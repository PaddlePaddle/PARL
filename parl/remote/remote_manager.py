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

from six.moves import queue
import threading
import time
import zmq
from parl.utils import logger, to_byte, to_str
from parl.remote import remote_constants
from parl.remote.remote_object import RemoteObject
"""
Two steps to build the communication with remote clients:
1. Create a RemoteManager;
2. Get remote objects by calling the function get_remote.

```python
    remote_manager = RemoteManager(port=[port])
    remote_obj = remote_manager.get_remote()
```

"""


class RemoteManager(object):
    """
    Base class for network communcation.
    """

    def __init__(self, port):
        """
        Args:
            port(int): a local port used for connections from remote clients.
        """
        self.zmq_context = zmq.Context()
        socket = self.zmq_context.socket(zmq.REP)
        socket.bind("tcp://*:{}".format(port))
        self.socket = socket
        self.socket.linger = 0

        self.remote_pool = queue.Queue()
        self.remote_latest_timestamp = {}

        t = threading.Thread(target=self._wait_for_connection)
        t.setDaemon(True)  # The thread will exit when main thread exited
        t.start()

        t = threading.Thread(target=self._check_remote_status)
        t.setDaemon(True)  # The thread will exit when main thread exited
        t.start()

    def _wait_for_connection(self):
        """
        A never-ending function keeps waiting for the connections from remote client.
        It will put an available remote object in an internel pool, and remote object
        can be obtained by calling `get_remote`.

        Note that this function has been called inside the `__init__` function.
        """
        remote_id = 0
        while True:
            try:
                message = self.socket.recv_multipart()
                tag = message[0]

                if tag == remote_constants.CONNECT_TAG:
                    self.socket.send_multipart([
                        remote_constants.NORMAL_TAG,
                        to_byte('{}'.format(remote_id))
                    ])
                    remote_client_address = to_str(message[1])
                    remote_obj = RemoteObject(remote_client_address,
                                              self.zmq_context)

                    self.remote_latest_timestamp[to_byte(
                        str(remote_id))] = time.time()

                    logger.info('[RemoteManager] Added a new remote object.')
                    self.remote_pool.put(remote_obj)

                    remote_id += 1

                elif tag == remote_constants.HEARTBEAT_TAG:
                    self.remote_latest_timestamp[message[1]] = time.time()
                    self.socket.send_multipart(
                        [remote_constants.NORMAL_TAG, b'Server is alive.'])
                else:
                    raise NotImplementedError()

            except zmq.ZMQError:
                logger.warning('Zmq error, exiting server.')
                break

    def _check_remote_status(self):
        while True:
            for remote_id in list(self.remote_latest_timestamp.keys()):
                if time.time() - self.remote_latest_timestamp[
                        remote_id] > 3 * remote_constants.HEARTBEAT_INTERVAL_S:
                    logger.error(
                        'Remote object {} is lost, please check if anything wrong happens in the remote client'
                        .format(remote_id))
                    self.remote_latest_timestamp.pop(remote_id)
            time.sleep(3 * remote_constants.HEARTBEAT_INTERVAL_S)

    def get_remote(self):
        """
        A blocking function to obtain a remote object.

        Returns:
            RemoteObject
        """
        return self.remote_pool.get()

    def close(self):
        """
        Close RemoteManager. 
        """

        self.zmq_context.destroy()
