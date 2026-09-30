#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""Authentication for xparl's trusted cluster boundary."""

import hashlib
import hmac
import os
import threading

import zmq
from zmq.auth.thread import ThreadAuthenticator
from zmq.utils import z85


def get_auth_token():
    token = os.environ.get('XPARL_AUTH_TOKEN', '')
    if len(token.encode('utf-8')) < 32:
        raise ValueError('Set XPARL_AUTH_TOKEN to a random secret of at least 32 bytes on every xparl node.')
    return token


def _curve_keys(token, role):
    secret = z85.encode(hashlib.sha256(('xparl-curve-v1:' + role + ':' + token).encode('utf-8')).digest())
    return zmq.curve_public(secret), secret


class _ClusterCredentials(object):
    def __init__(self, public_key):
        self.public_key = public_key

    def callback(self, domain, key):
        return domain == 'xparl' and hmac.compare_digest(key, self.public_key)


class SecureContext(zmq.Context):
    """A context that owns one CURVE authenticator and closes it with its sockets."""

    _xparl_token = None
    _xparl_authenticator = None
    _xparl_auth_lock = None

    def __init__(self, *args, **kwargs):
        token = get_auth_token()
        if not zmq.has('curve'):
            raise RuntimeError('xparl requires a libzmq build with CURVE support.')
        super(SecureContext, self).__init__(*args, **kwargs)
        self._xparl_token = token
        self._xparl_authenticator = None
        self._xparl_auth_lock = threading.Lock()

    def authenticate_server(self, socket):
        server_public, server_secret = _curve_keys(self._xparl_token, 'server')
        client_public, _ = _curve_keys(self._xparl_token, 'client')
        with self._xparl_auth_lock:
            if self._xparl_authenticator is None:
                authenticator = ThreadAuthenticator(self)
                authenticator.start()
                try:
                    authenticator.configure_curve_callback('xparl', _ClusterCredentials(client_public))
                except Exception:
                    authenticator.stop()
                    raise
                self._xparl_authenticator = authenticator
        socket.curve_publickey = server_public
        socket.curve_secretkey = server_secret
        socket.curve_server = True
        socket.zap_domain = b'xparl'

    def authenticate_client(self, socket):
        client_public, client_secret = _curve_keys(self._xparl_token, 'client')
        server_public, _ = _curve_keys(self._xparl_token, 'server')
        socket.curve_publickey = client_public
        socket.curve_secretkey = client_secret
        socket.curve_serverkey = server_public

    def _stop_authenticator(self):
        authenticator = getattr(self, '_xparl_authenticator', None)
        if authenticator is not None:
            authenticator.stop()
            self._xparl_authenticator = None

    def term(self):
        self._stop_authenticator()
        super(SecureContext, self).term()

    def destroy(self, linger=None):
        self._stop_authenticator()
        super(SecureContext, self).destroy(linger=linger)


def get_control_bind_host():
    """Remote control and HTTP exposure require an explicit bind address."""
    return os.environ.get('XPARL_BIND_HOST', '127.0.0.1')


def require_http_auth():
    """Protect monitor and log endpoints, including static resources."""
    from flask import request, Response
    token = get_auth_token()
    auth = request.authorization
    if auth is not None and auth.type == 'basic' and auth.username == 'xparl':
        if hmac.compare_digest((auth.password or '').encode('utf-8'), token.encode('utf-8')):
            return None
    return Response('Authentication required.', 401, {'WWW-Authenticate': 'Basic realm="xparl"'})
