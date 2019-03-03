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

import queue
import threading
import zmq
from parl.utils import logger, str2byte, byte2str
from parl.remote.remote_object import RemoteObject
from parl.remote.message import *

"""
2 steps to finish communication with remote clients.
1. Create a RemoteManager:
2. Get remote object after remote client connect to the RemoteManager

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

        t = threading.Thread(target=self._wait_for_connection)
        t.setDaemon(True) # The thread will exit when main thread exited
        t.start()

    def _wait_for_connection(self):
        """
        A never-ending function keeps waiting for the connections from remote client.
        It will put an available remote client into an internel client pool, and clients
        can be obtained by calling `get_client`.

        Note that this function has been called inside the `__init__` function.
        """
        while True:
            try:
                message = self.socket.recv_multipart()
                tag = message[0]

                if tag == CONNECT_TAG:
                    self.socket.send_multipart([NORMAL_TAG,
                        b'Connect server success.'])
                    client_info = byte2str(message[1])
                    remote_client_address, remote_client_id = client_info.split()
                    remote_obj = RemoteObject(remote_client_address, 
                            remote_client_id)
                    logger.info('[RemoteManager] Added a new remote object.')
                    self.remote_pool.put(remote_obj)
                elif tag == HEARTBEAT_TAG:
                    self.socket.send_multipart([NORMAL_TAG, b'Server is alive.'])
                else:
                    raise NotImplementedError()

            except zmq.ContextTerminated:
                logger.warning('Zmq context termnated, exiting server.')
                break


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
        
        self.socket.close()

        self.zmq_context.term()
