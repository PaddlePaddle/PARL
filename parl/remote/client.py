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
import time
import glob
import multiprocessing as mp

from parl.utils import to_str, to_byte, get_ip_address, logger, isnotebook
from parl.remote.utils import get_subfiles_recursively
from parl.remote import remote_constants
from parl.remote.grpc_heartbeat import HeartbeatServerThread, HeartbeatServerProcess
from parl.remote.utils import get_version


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
            master_addr (str): IP address of the master node.
            job_heartbeat_server_addr(str): Server address for heartbeat detection from jobs.
            process_id (str): Process id in which client is created. Should use os.getpid() to get the process id.
            distributed_files (list): A list of files to be distributed at all remote instances(e,g. the configuration
                                      file for initialization) .
        """
        self.dead_job_queue = mp.Queue()
        self.client_is_alive = mp.Value('i', True)
        self._create_heartbeat_server()
        th = threading.Thread(target=self._update_job_status, args=(self.dead_job_queue, ))
        th.setDaemon(True)
        th.start()
        self.master_address = master_address
        self.process_id = process_id
        self.ctx = zmq.Context()
        self.lock = threading.Lock()
        self.log_monitor_url = None
        self.threads = []
        self.executable_path = self.get_executable_path()
        self._create_sockets(master_address)
        self.connected_to_master = True
        self.check_env_consistency()
        self.instance_count = 0
        self.instance_id_to_job = dict()

        thread = threading.Thread(target=self._update_client_status_to_master)
        thread.setDaemon(True)
        thread.start()
        self.threads.append(thread)

        self.pyfiles = self.read_local_files(distributed_files)

    def destroy(self):
        """Destructor function"""
        self.connected_to_master = False
        self.dead_job_queue.put('exit')
        self.master_heartbeat_thread.exit()
        for th in self.threads:
            th.join()
        self.ctx.destroy()
        self.client_is_alive.value = False
        self.job_heartbeat_process.join()

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
                raise ValueError("no local file is matched with '{}', please check your input".format(distributed_file))
            for pathname in parsed_list:
                if os.path.isdir(pathname):
                    pythonfiles, otherfiles, emptysubfolders = get_subfiles_recursively(pathname)
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
        code_files = filter(lambda x: x.endswith('.py'), os.listdir(main_folder))

        for file_name in code_files:
            file_path = os.path.join(main_folder, file_name)
            assert os.path.exists(file_path)
            with open(file_path, 'rb') as code_file:
                code = code_file.read()
                pyfiles['python_files'][file_name] = code

        for file_name in set(user_files):
            assert os.path.exists(file_name)
            assert not os.path.isabs(file_name), "[XPARL] Please do not distribute a file with absolute path."
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
        self.submit_job_socket.setsockopt(zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)
        self.submit_job_socket.connect("tcp://{}".format(master_address))
        self.start_time = time.time()

        def master_heartbeat_exit_callback_func():
            logger.warning("[Client] Cannot connect to the master. " "Please check if it is still alive.")
            logger.warning("Client exit replying heartbeat for master.")
            self.connected_to_master = False

        self.master_heartbeat_thread = HeartbeatServerThread(
            heartbeat_exit_callback_func=master_heartbeat_exit_callback_func)
        self.master_heartbeat_thread.setDaemon(True)
        self.master_heartbeat_thread.start()
        self.reply_master_heartbeat_address = self.master_heartbeat_thread.get_address()
        self.threads.append(self.master_heartbeat_thread)

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
            self.connected_to_master = False
            raise Exception("Client can not connect to the master, please "
                            "check if master is started and ensure the input "
                            "address {} is correct.".format(master_address))

    def _update_job_status(self, dead_job_queue):
        while True:
            instance_id = dead_job_queue.get()
            # the client calls the destroy function
            if isinstance(instance_id, str) and instance_id == 'exit':
                break
            logger.error("[Client] lost connection with a remote instance. ID: {}".format(instance_id))
            job_is_alive = self.instance_id_to_job[instance_id]
            job_is_alive.value = False

    def check_env_consistency(self):
        '''Verify that the parl & python version as well as some other packages in 'worker' process
            matches that of the 'master' process'''
        self.submit_job_socket.send_multipart([remote_constants.CHECK_VERSION_TAG])
        message = self.submit_job_socket.recv_multipart()
        tag = message[0]
        if tag == remote_constants.NORMAL_TAG:
            client_parl_version = parl.__version__
            client_python_version_major = str(sys.version_info.major)
            client_python_version_minor = str(sys.version_info.minor)
            assert client_parl_version == to_str(message[1]) and client_python_version_major == to_str(message[2])\
                and client_python_version_minor == to_str(message[3]),\
                '''Version mismatch: the 'master' is of version 'parl={}, python={}.{}'. However, \
                'parl={}, python={}.{}'is provided in your environment.'''.format(
                        to_str(message[1]), to_str(message[2]), to_str(message[3]),
                        client_parl_version, client_python_version_major, client_python_version_minor
                    )
            client_pyarrow_version = str(get_version('pyarrow'))
            master_pyarrow_version = to_str(message[4])
            if client_pyarrow_version != master_pyarrow_version:
                if master_pyarrow_version == 'None':
                    error_message = """"pyarrow" is provided in your current enviroment, however, it is not \
found in "master"'s environment. To use "pyarrow" for serialization, please install \
"pyarrow={}" in "master"'s environment!""".format(client_pyarrow_version)
                elif client_pyarrow_version == 'None':
                    error_message = """"pyarrow" is provided in "master"'s enviroment, however, it is not \
found in your current environment. To use "pyarrow" for serialization, please install \
"pyarrow={}" in your current environment!""".format(master_pyarrow_version)
                else:
                    error_message = '''Version mismatch: the 'master' is of version 'pyarrow={}'. However, \
'pyarrow={}'is provided in your current environment.'''.format(master_pyarrow_version, client_pyarrow_version)
                raise Exception(error_message)
        else:
            raise NotImplementedError

    def _update_client_status_to_master(self):
        while self.connected_to_master:
            elapsed_time = datetime.timedelta(seconds=int(time.time() - self.start_time))
            client_status = {
                'file_path': self.executable_path,
                'actor_num': self.actor_num.value,
                'time': str(elapsed_time),
                'log_monitor_url': self.log_monitor_url
            }

            self.lock.acquire()
            try:
                self.submit_job_socket.send_multipart([
                    remote_constants.CLIENT_STATUS_UPDATE_TAG,
                    to_byte(self.reply_master_heartbeat_address),
                    cloudpickle.dumps(client_status)
                ])
                message = self.submit_job_socket.recv_multipart()
            except zmq.error.Again as e:
                self.connected_to_master = False
            finally:
                self.lock.release()

            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

    def _check_job(self, job_ping_address, max_memory, gpu):
        """ 
        Check if this job is still alive before establishing connection with it.
        Return: instance_id (int): an unique isntance id. -1 if the job is not ready for connection.
        """
        # job_ping_socket: sends ping signal to job
        job_ping_socket = self.ctx.socket(zmq.REQ)
        job_ping_socket.linger = 0
        job_ping_socket.setsockopt(zmq.RCVTIMEO, int(0.9 * 1000))
        job_ping_socket.connect("tcp://" + job_ping_address)
        instance_id = self._generate_instance_id()
        try:
            job_ping_socket.send_multipart([
                remote_constants.HEARTBEAT_TAG,
                to_byte(self.job_heartbeat_server_addr),
                to_byte(str(max_memory)),
                to_byte(gpu),
                to_byte(instance_id)
            ], )
            job_ping_socket.recv_multipart()
        except zmq.error.Again:
            logger.error(
                "[Client] connects to a finished job, will try again, job_ping_address:{}".format(job_ping_address))
            instance_id = -1
        finally:
            job_ping_socket.close(0)
        return instance_id

    def _create_heartbeat_server(self):
        """ Create the grpc-based heartbeat server at the subprocess.
        """
        job_heartbeat_port = mp.Value('i', 0)
        self.actor_num = mp.Value('i', 0)
        self.job_heartbeat_process = HeartbeatServerProcess(job_heartbeat_port, self.actor_num, 
                                         self.client_is_alive, self.dead_job_queue)
        self.job_heartbeat_process.daemon = True
        self.job_heartbeat_process.start()
        assert job_heartbeat_port.value != 0, "fail to initialize heartbeat server for jobs."
        self.job_heartbeat_server_addr = "{}:{}".format(get_ip_address(), job_heartbeat_port.value)

    def _generate_instance_id(self):
        """Return an unique instance id for the remote instance"""
        self.instance_count += 1
        unique_id = f"{self.instance_count:05}"
        return unique_id

    def submit_job(self, max_memory, n_gpu, job_is_alive):
        """Send a job to the Master node.

        When a `@parl.remote_class` object is created, the global client
        sends a job to the master node. Then the master node will allocate
        a vacant job from its job pool to the remote object.

        Args:
            max_memory (float): Maximum memory (MB) can be used by each remote
                                instance, the unit is in MB and default value is
                                none(unlimited).
            n_gpu (int): Number of GPUs can used in this remote instance.
        Returns:
            An ``InitializedJob`` that has information about available job address.
        """
        if self.connected_to_master:

            while True:
                self.lock.acquire()
                n_cpu = 0 if n_gpu > 0 else 1
                self.submit_job_socket.send_multipart([
                    remote_constants.CLIENT_SUBMIT_TAG,
                    to_byte(self.reply_master_heartbeat_address),
                    to_byte(self.client_id),
                    to_byte(str(n_cpu)),
                    to_byte(str(n_gpu))
                ])
                message = self.submit_job_socket.recv_multipart()
                self.lock.release()
                tag = message[0]
                if tag == remote_constants.NORMAL_TAG:
                    job_info = cloudpickle.loads(message[1])
                    job_ping_address = job_info.ping_heartbeat_address

                    self.lock.acquire()
                    instance_id = self._check_job(job_ping_address, max_memory, job_info.allocated_gpu.gpu)
                    self.lock.release()
                    if instance_id != -1:
                        self.instance_id_to_job[instance_id] = job_is_alive
                        return job_info
                # no vacant CPU resources, cannot submit a new job
                elif tag == remote_constants.CPU_TAG:
                    # wait 1 second to avoid requesting in a high frequency.
                    time.sleep(1)
                    return None
                # no vacant GPU resources, cannot submit a new job
                elif tag == remote_constants.GPU_TAG:
                    # wait 5 second to avoid requesting in a high frequency.
                    time.sleep(1)
                    return None
                elif tag == remote_constants.REJECT_GPU_JOB_TAG:
                    error_message = "[Client] Request fails. It is not allowed to request CPU resource from a GPU cluster."
                    logger.error(error_message)
                    raise Exception(error_message)
                elif tag == remote_constants.REJECT_CPU_JOB_TAG:
                    error_message = "[Client] Request fails. It is not allowed to request GPU resource from a CPU cluster."
                    logger.error(error_message)
                    raise Exception(error_message)
                elif tag == remote_constants.REJECT_INVALID_GPU_JOB_TAG:
                    error_message = "[Client] request {} GPUs, but rejected.".format(n_gpu)
                    logger.error(error_message)
                    raise Exception(error_message)
                else:
                    raise NotImplementedError
        else:
            raise Exception("Client can not submit job to the master, please check if master is connected.")
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
    assert isinstance(distributed_files, list), "`distributed_files` should be a list."

    global GLOBAL_CLIENT
    addr = master_address.split(":")[0]
    cur_process_id = os.getpid()
    if GLOBAL_CLIENT is None:
        GLOBAL_CLIENT = Client(master_address, cur_process_id, distributed_files)
    else:
        if GLOBAL_CLIENT.process_id != cur_process_id:
            GLOBAL_CLIENT = Client(master_address, cur_process_id, distributed_files)
    logger.info("Remote actors log url: {}".format(GLOBAL_CLIENT.log_monitor_url))


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
        GLOBAL_CLIENT.destroy()
        GLOBAL_CLIENT = None
        logger.info("The client is disconneced to the master node.")
    else:
        logger.info("No client to be released. Please make sure that you have called `parl.connect`")
