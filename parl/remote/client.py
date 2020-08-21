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
import datetime
import os
import socket
import sys
import threading
import zmq
import parl
from parl.utils import to_str, to_byte, get_ip_address, logger, isnotebook
from parl.remote.utils import get_subfiles_recursively
from parl.remote import remote_constants
import time
import glob


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
        executable_path (str): File path of the executable python script.
        start_time (time): A timestamp to record the start time of the program.

    """

    def __init__(self, master_address, process_id, distributed_files=[]):
        """
        Args:
            master_addr (str): ip address of the master node.
            process_id (str): id of the process that created the Client. 
                              Should use os.getpid() to get the process id.
            distributed_files (list): A list of files to be distributed at all
                                      remote instances(e,g. the configuration
                                      file for initialization) .
        """
        self.master_address = master_address
        self.process_id = process_id
        self.ctx = zmq.Context()
        self.lock = threading.Lock()
        self.heartbeat_socket_initialized = threading.Event()
        self.master_is_alive = True
        self.client_is_alive = True
        self.log_monitor_url = None

        self.executable_path = self.get_executable_path()

        self.actor_num = 0

        self._create_sockets(master_address)
        self.check_version()
        self.pyfiles = self.read_local_files(distributed_files)

    def get_executable_path(self):
        """Return current executable path."""
        mod = sys.modules['__main__']
        if hasattr(mod, '__file__'):
            executable_path = os.path.abspath(mod.__file__)
        else:
            executable_path = os.getcwd()
        executable_path = executable_path[:executable_path.rfind('/')]
        return executable_path

    def read_local_files(self, distributed_files=[]):
        """Read local python code and store them in a dictionary, which will
        then be sent to the job.

        Args:
            distributed_files (list): A list of files to be distributed at all
                                      remote instances(e,g. the configuration
                                      file for initialization) . RegExp of file
                                      names is supported. 
                                      e.g. 
                                          distributed_files = ['./*.npy', './test*']
                                                                             
        Returns:
            A cloudpickled dictionary containing the python code in current
            working directory.
        """
        pyfiles = dict()
        pyfiles['python_files'] = {}
        pyfiles['other_files'] = {}

        user_files = []
        user_empty_subfolders = []

        for distributed_file in distributed_files:
            parsed_list = glob.glob(distributed_file)
            if not parsed_list:
                raise ValueError(
                    "no local file is matched with '{}', please check your input"
                    .format(distributed_file))
            for pathname in parsed_list:
                if os.path.isdir(pathname):
                    pythonfiles, otherfiles, emptysubfolders = get_subfiles_recursively(
                        pathname)
                    user_files.extend(pythonfiles)
                    user_files.extend(otherfiles)
                    user_empty_subfolders.extend(emptysubfolders)
                else:
                    user_files.append(pathname)

        if isnotebook():
            main_folder = './'
        else:
            main_file = sys.argv[0]
            main_folder = './'
            sep = os.sep
            if sep in main_file:
                main_folder = sep.join(main_file.split(sep)[:-1])
        code_files = filter(lambda x: x.endswith('.py'),
                            os.listdir(main_folder))

        for file_name in code_files:
            file_path = os.path.join(main_folder, file_name)
            assert os.path.exists(file_path)
            with open(file_path, 'rb') as code_file:
                code = code_file.read()
                pyfiles['python_files'][file_name] = code

        for file_name in set(user_files):
            assert os.path.exists(file_name)
            assert not os.path.isabs(
                file_name
            ), "[XPARL] Please do not distribute a file with absolute path."
            with open(file_name, 'rb') as f:
                content = f.read()
                pyfiles['other_files'][file_name] = content

        pyfiles['empty_subfolders'] = set(user_empty_subfolders)
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
        self.start_time = time.time()
        thread = threading.Thread(target=self._reply_heartbeat)
        thread.setDaemon(True)
        thread.start()
        self.heartbeat_socket_initialized.wait()

        self.client_id = self.reply_master_heartbeat_address.replace(':', '_') + \
                            '_' + str(int(time.time()))

        # check if the master is connected properly
        try:
            self.submit_job_socket.send_multipart([
                remote_constants.CLIENT_CONNECT_TAG,
                to_byte(self.reply_master_heartbeat_address),
                to_byte(socket.gethostname()),
                to_byte(self.client_id),
            ])
            message = self.submit_job_socket.recv_multipart()
            self.log_monitor_url = to_str(message[1])
        except zmq.error.Again as e:
            logger.warning("[Client] Can not connect to the master, please "
                           "check if master is started and ensure the input "
                           "address {} is correct.".format(master_address))
            self.master_is_alive = False
            raise Exception("Client can not connect to the master, please "
                            "check if master is started and ensure the input "
                            "address {} is correct.".format(master_address))

    def check_version(self):
        '''Verify that the parl & python version in 'client' process matches that of the 'master' process'''
        self.submit_job_socket.send_multipart(
            [remote_constants.CHECK_VERSION_TAG])
        message = self.submit_job_socket.recv_multipart()
        tag = message[0]
        if tag == remote_constants.NORMAL_TAG:
            client_parl_version = parl.__version__
            client_python_version = str(sys.version_info.major)
            assert client_parl_version == to_str(message[1]) and client_python_version == to_str(message[2]),\
                '''Version mismatch: the 'master' is of version 'parl={}, python={}'. However, 
                'parl={}, python={}'is provided in your environment.'''.format(
                        to_str(message[1]), to_str(message[2]),
                        client_parl_version, client_python_version
                    )
        else:
            raise NotImplementedError

    def _reply_heartbeat(self):
        """Reply heartbeat signals to the master node."""

        socket = self.ctx.socket(zmq.REP)
        socket.linger = 0
        socket.setsockopt(zmq.RCVTIMEO,
                          remote_constants.HEARTBEAT_RCVTIMEO_S * 1000)
        reply_master_heartbeat_port =\
            socket.bind_to_random_port(addr="tcp://*")
        self.reply_master_heartbeat_address = "{}:{}".format(
            get_ip_address(), reply_master_heartbeat_port)
        self.heartbeat_socket_initialized.set()
        connected = False
        while self.client_is_alive and self.master_is_alive:
            try:
                message = socket.recv_multipart()
                elapsed_time = datetime.timedelta(
                    seconds=int(time.time() - self.start_time))
                socket.send_multipart([
                    remote_constants.HEARTBEAT_TAG,
                    to_byte(self.executable_path),
                    to_byte(str(self.actor_num)),
                    to_byte(str(elapsed_time)),
                    to_byte(str(self.log_monitor_url)),
                ])  # TODO: remove additional information
            except zmq.error.Again as e:
                if connected:
                    logger.warning("[Client] Cannot connect to the master."
                                   "Please check if it is still alive.")
                else:
                    logger.warning(
                        "[Client] Cannot connect to the master."
                        "Please check the firewall between client and master.(e.g., ping the master IP)"
                    )
                self.master_is_alive = False
        socket.close(0)
        logger.warning("Client exit replying heartbeat for master.")

    def _check_and_monitor_job(self, job_heartbeat_address,
                               ping_heartbeat_address, max_memory):
        """ Sometimes the client may receive a job that is dead, thus 
        we have to check if this job is still alive before adding it to the `actor_num`.
        """
        # job_heartbeat_socket: sends heartbeat signal to job
        job_heartbeat_socket = self.ctx.socket(zmq.REQ)
        job_heartbeat_socket.linger = 0
        job_heartbeat_socket.setsockopt(zmq.RCVTIMEO, int(0.9 * 1000))
        job_heartbeat_socket.connect("tcp://" + ping_heartbeat_address)
        try:
            job_heartbeat_socket.send_multipart(
                [remote_constants.HEARTBEAT_TAG,
                 to_byte(str(max_memory))])
            job_heartbeat_socket.recv_multipart()
        except zmq.error.Again:
            job_heartbeat_socket.close(0)
            logger.error(
                "[Client] connects to a finished job, will try again, ping_heartbeat_address:{}"
                .format(ping_heartbeat_address))
            return False
        job_heartbeat_socket.disconnect("tcp://" + ping_heartbeat_address)
        job_heartbeat_socket.connect("tcp://" + job_heartbeat_address)
        job_heartbeat_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)

        # a thread for sending heartbeat signals to job
        thread = threading.Thread(
            target=self._create_job_monitor, args=(job_heartbeat_socket, ))
        thread.setDaemon(True)
        thread.start()
        return True

    def _create_job_monitor(self, job_heartbeat_socket):
        """Send heartbeat signals to check target's status"""

        job_is_alive = True
        while job_is_alive and self.client_is_alive:
            try:
                job_heartbeat_socket.send_multipart(
                    [remote_constants.HEARTBEAT_TAG])
                job_message = job_heartbeat_socket.recv_multipart()
                stop_job = to_str(job_message[1])
                job_address = to_str(job_message[2])

                if stop_job == 'True':
                    logger.error(
                        'Job {} exceeds max memory usage, will stop this job.'.
                        format(job_address))
                    self.lock.acquire()
                    self.actor_num -= 1
                    self.lock.release()
                    job_is_alive = False
                else:
                    time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

            except zmq.error.Again as e:
                job_is_alive = False
                self.lock.acquire()
                self.actor_num -= 1
                logger.error(
                    '[xparl] lost connection with a job, current actor num: {}'
                    .format(self.actor_num))
                self.lock.release()

            except zmq.error.ZMQError as e:
                break

        job_heartbeat_socket.close(0)

    def submit_job(self, max_memory):
        """Send a job to the Master node.

        When a `@parl.remote_class` object is created, the global client
        sends a job to the master node. Then the master node will allocate
        a vacant job from its job pool to the remote object.

        Args:
            max_memory (float): Maximum memory (MB) can be used by each remote
                                instance, the unit is in MB and default value is
                                none(unlimited).

        Returns:
            job_address(str): IP address of the job. None if there is no available CPU in the cluster.
        """
        if self.master_is_alive:

            while True:
                # A lock to prevent multiple actors from submitting job at the same time.
                self.lock.acquire()
                self.submit_job_socket.send_multipart([
                    remote_constants.CLIENT_SUBMIT_TAG,
                    to_byte(self.reply_master_heartbeat_address),
                    to_byte(self.client_id),
                ])
                message = self.submit_job_socket.recv_multipart()
                self.lock.release()

                tag = message[0]

                if tag == remote_constants.NORMAL_TAG:
                    job_address = to_str(message[1])
                    job_heartbeat_address = to_str(message[2])
                    ping_heartbeat_address = to_str(message[3])

                    check_result = self._check_and_monitor_job(
                        job_heartbeat_address, ping_heartbeat_address,
                        max_memory)
                    if check_result:
                        self.lock.acquire()
                        self.actor_num += 1
                        self.lock.release()
                        return job_address

                # no vacant CPU resources, cannot submit a new job
                elif tag == remote_constants.CPU_TAG:
                    job_address = None
                    # wait 1 second to avoid requesting in a high frequency.
                    time.sleep(1)
                    return job_address
                else:
                    raise NotImplementedError
        else:
            raise Exception("Client can not submit job to the master, "
                            "please check if master is connected.")
        return None


GLOBAL_CLIENT = None


def connect(master_address, distributed_files=[]):
    """Create a global client which connects to the master node.

    .. code-block:: python

        parl.connect(master_address='localhost:1234')

    Args:
        master_address (str): The address of the Master node to connect to.
        distributed_files (list): A list of files to be distributed at all 
                                  remote instances(e,g. the configuration
                                  file for initialization) .

    Raises:
        Exception: An exception is raised if the master node is not started.
    """

    assert len(master_address.split(":")) == 2, "Please input address in " +\
        "{ip}:{port} format"
    global GLOBAL_CLIENT
    addr = master_address.split(":")[0]
    cur_process_id = os.getpid()
    if GLOBAL_CLIENT is None:
        GLOBAL_CLIENT = Client(master_address, cur_process_id,
                               distributed_files)
    else:
        if GLOBAL_CLIENT.process_id != cur_process_id:
            GLOBAL_CLIENT = Client(master_address, cur_process_id,
                                   distributed_files)
    logger.info("Remote actors log url: {}".format(
        GLOBAL_CLIENT.log_monitor_url))


def get_global_client():
    """Get the global client.

    To support process-based programming, we will create a new global client in the new process.

    Returns:
        The global client.
    """
    global GLOBAL_CLIENT
    assert GLOBAL_CLIENT is not None, "Cannot get the client to submit the" +\
        " job, have you connected to the cluster by calling " +\
        "parl.connect(master_ip, master_port)?"

    cur_process_id = os.getpid()
    if GLOBAL_CLIENT.process_id != cur_process_id:
        GLOBAL_CLIENT = Client(GLOBAL_CLIENT.master_address, cur_process_id)
    return GLOBAL_CLIENT


def disconnect():
    """Disconnect the global client from the master node."""
    global GLOBAL_CLIENT
    if GLOBAL_CLIENT is not None:
        GLOBAL_CLIENT.client_is_alive = False
        GLOBAL_CLIENT = None
    else:
        logger.info(
            "No client to be released. Please make sure that you have called `parl.connect`"
        )
