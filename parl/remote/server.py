#!/usr/bin/env python
# coding=utf8
# File: server.py
import zmq
import threading
import queue
from decorator import virtual
from parl.utils import logger
"""
3 steps to finish communication with remote clients.
1. Create a server:
2. Declare the type of remote client
3. Get remote clients

```python
    server = Server()
    server.bind(Simulator) 
    remote_client = server.get_client()
```

"""


class Server(object):
    """
    Base class for network communcation.
    """

    def __init__(self, port):
        """
        Args:
            port(int): a local port used for network communication.
        """
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:{}".format(port))
        self.socket = socket
        self.pool = queue.Queue()
        self.cls = None
        t = threading.Thread(target=self.wait_for_connection)
        t.start()

    def wait_for_connection(self):
        """
        A never-ending function keeps waiting for the connection for remote client.
        It will put an available remote client into an internel client pool, and clients
        can be obtained by calling `get_client`.

        Note that this function has been called inside the `__init__` function.
        """
        while True:
            client_info = self.socket.recv_string()
            client_id = client_info.split(' ')[1]
            new_client = virtual(self.cls, location='server')()
            self.socket.send_string('Hello World! Client:{}'.format(client_id))
            new_client.connect_client(client_info)
            self.pool.put(new_client)

    def get_client(self):
        """
        A blocking function to obtain a connected client.

        Returns:
            remote_client(self.cls): a **remote** instance that has all functions as the real one.
        """
        return self.pool.get()

    def register_client(self, cls):
        """
        Declare the type of remote class.
        Let the server know which class to use as a remote client.

        Args:
            cls(Class): A class decorated by @virtual.
        """
        self.cls = cls
