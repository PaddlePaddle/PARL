#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cloudpickle
import os
import threading
import zmq
from parl.utils import to_str, to_byte, get_ip_address, logger
from parl.remote import remote_constants
import time


class Client(object):
    """Base class for the remote client.

    For each training task, there is a global client in the cluster which
    submits jobs to the master node. Different `@parl.remote_class` objects
    connect to the same global client in a training task.

    Attributes:
        submit_job_socket (zmq.Context.socket): A socket which submits job to
                                                the master node.
        pyfiles (bytes): A serialized dictionary containing the code of python
                         files in local working directory.

    """

    def __init__(self, master_address):
        """
        Args:
            master_addr (str): ip address of the master node.
        """
        self.ctx = zmq.Context()
        self.lock = threading.Lock()
        self.heartbeat_socket_initialized = threading.Event()
        self.master_is_alive = True
        self.client_is_alive = True

        self._create_sockets(master_address)
        self.pyfiles = self.read_local_files()

    def read_local_files(self):
        """Read local python code and store them in a dictionary, which will
        then be sent to the job.

        Returns:
            A cloudpickled dictionary containing the python code in current
            working directory.
        """
        pyfiles = dict()
        for file in os.listdir('./'):
            if file.endswith('.py'):
                with open(file, 'rb') as code_file:
                    code = code_file.read()
                    pyfiles[file] = code
        return cloudpickle.dumps(pyfiles)

    def _create_sockets(self, master_address):
        """ Each client has 1 sockets as start:

        (1) submit_job_socket: submits jobs to master node.
        """

        # submit_job_socket: submits job to master
        self.submit_job_socket = self.ctx.socket(zmq.REQ)
        self.submit_job_socket.linger = 0
        self.submit_job_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)
        self.submit_job_socket.connect("tcp://{}".format(master_address))

        thread = threading.Thread(target=self._reply_heartbeat)
        thread.setDaemon(True)
        thread.start()
        self.heartbeat_socket_initialized.wait()

        # check if the master is connected properly
        try:
            self.submit_job_socket.send_multipart([
                remote_constants.CLIENT_CONNECT_TAG,
                to_byte(self.heartbeat_master_address)
            ])
            _ = self.submit_job_socket.recv_multipart()
        except zmq.error.Again as e:
            logger.warning("[Client] Can not connect to the master, please "
                           "check if master is started and ensure the input "
                           "address {} is correct.".format(master_address))
            self.master_is_alive = False
            raise Exception("Client can not connect to the master, please "
                            "check if master is started and ensure the input "
                            "address {} is correct.".format(master_address))

    def _reply_heartbeat(self):
        """Reply heartbeat signals to the Master node."""

        socket = self.ctx.socket(zmq.REP)
        socket.linger = 0
        socket.setsockopt(zmq.RCVTIMEO,
                          remote_constants.HEARTBEAT_RCVTIMEO_S * 1000)
        heartbeat_master_port =\
            socket.bind_to_random_port(addr="tcp://*")
        self.heartbeat_master_address = "{}:{}".format(get_ip_address(),
                                                       heartbeat_master_port)
        self.heartbeat_socket_initialized.set()
        while self.client_is_alive and self.master_is_alive:
            try:
                message = socket.recv_multipart()
                socket.send_multipart([remote_constants.HEARTBEAT_TAG])

            except zmq.error.Again as e:
                logger.warning("[Client] Cannot connect to the master."
                               "Please check if it is still alive.")
                self.master_is_alive = False
        socket.close(0)
        logger.warning("Client exit replying heartbeat for master.")

    def submit_job(self):
        """Send a job to the Master node.

        When a `@parl.remote_class` object is created, the global client
        sends a job to the master node. Then the master node will allocate
        a vacant job from its job pool to the remote object.

        Returns:
            IP address of the job.
        """
        if self.master_is_alive:

            # A lock to prevent multiple actor submit job at the same time.
            self.lock.acquire()
            self.submit_job_socket.send_multipart([
                remote_constants.CLIENT_SUBMIT_TAG,
                to_byte(self.heartbeat_master_address)
            ])
            message = self.submit_job_socket.recv_multipart()
            self.lock.release()

            tag = message[0]

            if tag == remote_constants.NORMAL_TAG:
                job_address = to_str(message[1])

            # no vacant CPU resources, can not submit a new job
            elif tag == remote_constants.CPU_TAG:
                job_address = None
                # wait 1 second to avoid requesting in a high frequency.
                time.sleep(1)
            else:
                raise NotImplementedError
        else:
            raise Exception("Client can not submit job to the master, "
                            "please check if master is connected.")
        return job_address


GLOBAL_CLIENT = None


def connect(master_address):
    """Create a global client which connects to the master node.

    .. code-block:: python

        parl.connect(master_address='localhost:1234')

    Args:
        master_address (str): The address of the Master node to connect to.

    Raises:
        Exception: An exception is raised if the master node is not started.
    """

    assert len(master_address.split(":")) == 2, "please input address in " +\
        "{ip}:{port} format"
    global GLOBAL_CLIENT
    if GLOBAL_CLIENT is None:
        GLOBAL_CLIENT = Client(master_address)


def get_global_client():
    """Get the global client.

    Returns:
        The global client.
    """
    global GLOBAL_CLIENT
    assert GLOBAL_CLIENT is not None, "Cannot get the client to submit the" +\
        " job, have you connected to the cluster by calling " +\
        "parl.connect(master_ip, master_port)?"
    return GLOBAL_CLIENT


def disconnect():
    """Disconnect the global client from the master node."""
    global GLOBAL_CLIENT
    GLOBAL_CLIENT.client_is_alive = False
    GLOBAL_CLIENT = None
