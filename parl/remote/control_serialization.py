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
"""Data-only JSON for cluster metadata. Never fall back to pickle."""

import json
import math
from collections import deque

from parl.remote.message import AllocatedCpu, AllocatedGpu, InitializedJob, InitializedWorker

MAX_CONTROL_BYTES = 16 * 1024 * 1024
MAX_DEPTH = 32
_RECORD_TYPES = {cls.__name__: cls for cls in (AllocatedCpu, AllocatedGpu, InitializedJob, InitializedWorker)}
_FIELDS = {
    AllocatedCpu: {
        'worker_address': str,
        'n_cpu': int
    },
    AllocatedGpu: {
        'worker_address': str,
        'gpu': str
    },
    InitializedJob: {
        'job_address': str,
        'worker_heartbeat_address': str,
        'ping_heartbeat_address': str,
        'worker_address': (str, type(None)),
        'pid': int,
        'is_alive': bool,
        'job_id': (str, type(None)),
        'log_server_address': (str, type(None)),
        'allocated_cpu': (AllocatedCpu, type(None)),
        'allocated_gpu': (AllocatedGpu, type(None)),
        'instance_id': (str, int, type(None))
    },
    InitializedWorker: {
        'worker_address': str,
        'initialized_jobs': list,
        'allocated_cpu': AllocatedCpu,
        'allocated_gpu': AllocatedGpu,
        'hostname': str
    }
}


def _validate_record(value):
    schema = _FIELDS[type(value)]
    if set(vars(value)) != set(schema):
        raise ValueError('Invalid control record fields.')
    for name, allowed in schema.items():
        allowed = allowed if isinstance(allowed, tuple) else (allowed, )
        if type(getattr(value, name)) not in allowed:
            raise ValueError('Invalid control record field type.')
    if type(value) is InitializedWorker and any(type(job) is not InitializedJob for job in value.initialized_jobs):
        raise ValueError('Invalid worker job list.')
    if type(value) is AllocatedCpu and value.n_cpu < 0:
        raise ValueError('Negative CPU allocation.')


def _encode(value, depth=0):
    if depth > MAX_DEPTH:
        raise ValueError('Control metadata is too deeply nested.')
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple, deque)):
        return [_encode(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value) or '__xparl_type__' in value:
            raise ValueError('Invalid control dictionary keys.')
        return {key: _encode(item, depth + 1) for key, item in value.items()}
    if type(value) in _FIELDS:
        _validate_record(value)
        return {'__xparl_type__': type(value).__name__, 'fields': _encode(vars(value), depth + 1)}
    raise ValueError('Unsupported control metadata type.')


def _decode(value, depth=0):
    if depth > MAX_DEPTH:
        raise ValueError('Control metadata is too deeply nested.')
    if type(value) is list:
        return [_decode(item, depth + 1) for item in value]
    if type(value) is dict:
        if '__xparl_type__' in value:
            name = value['__xparl_type__']
            if type(name) is not str or name not in _RECORD_TYPES or set(value) != {'__xparl_type__', 'fields'}:
                raise ValueError('Unknown control record.')
            fields = _decode(value['fields'], depth + 1)
            if type(fields) is not dict:
                raise ValueError('Invalid control record.')
            record = object.__new__(_RECORD_TYPES[name])
            record.__dict__.update(fields)
            _validate_record(record)
            return record
        return {key: _decode(item, depth + 1) for key, item in value.items()}
    if value is None or type(value) in (str, int, bool) or (type(value) is float and math.isfinite(value)):
        return value
    raise ValueError('Invalid control metadata value.')


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key.')
        value[key] = item
    return value


def dumps(value):
    data = json.dumps(_encode(value), allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(data) > MAX_CONTROL_BYTES:
        raise ValueError('Control metadata is too large.')
    return data


def loads(data, expected_type=dict):
    if type(data) is not bytes or len(data) > MAX_CONTROL_BYTES:
        raise ValueError('Invalid control metadata size or type.')
    try:
        value = _decode(json.loads(data.decode('utf-8'), object_pairs_hook=_unique_object))
    except (RecursionError, UnicodeError) as error:
        raise ValueError('Invalid control JSON.') from error
    if type(value) is not expected_type:
        raise ValueError('Unexpected control metadata type.')
    return value


def loads_status(data, worker=False):
    value = loads(data)
    if worker:
        schema = {
            'vacant_memory': (int, float),
            'used_memory': (int, float),
            'vacant_gpu_memory': (int, float),
            'used_gpu_memory': (int, float),
            'load_time': (str, ),
            'load_value': (int, float)
        }
    else:
        schema = {'file_path': (str, ), 'actor_num': (int, ), 'time': (str, ), 'log_monitor_url': (str, )}
    if set(value) != set(schema) or any(type(value[key]) not in types for key, types in schema.items()):
        raise ValueError('Invalid status fields.')
    return value
