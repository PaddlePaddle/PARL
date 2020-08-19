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

CPU_TAG = b'[CPU]'
CONNECT_TAG = b'[CONNECT]'
HEARTBEAT_TAG = b'[HEARTBEAT]'
KILLJOB_TAG = b'[KILLJOB]'
MONITOR_TAG = b'[MONITOR]'
STATUS_TAG = b'[STATUS]'

WORKER_CONNECT_TAG = b'[WORKER_CONNECT]'
WORKER_INITIALIZED_TAG = b'[WORKER_INITIALIZED]'
CLIENT_CONNECT_TAG = b'[CLIENT_CONNECT]'
CLIENT_SUBMIT_TAG = b'[CLIENT_SUBMIT]'
SEND_FILE_TAG = b'[SEND_FILE]'
SUBMIT_JOB_TAG = b'[SUBMIT_JOB]'
NEW_JOB_TAG = b'[NEW_JOB]'

CHECK_VERSION_TAG = b'[CHECK_VERSION]'
INIT_OBJECT_TAG = b'[INIT_OBJECT]'
CALL_TAG = b'[CALL]'
GET_ATTRIBUTE_TAG = b'[GET_ATTRIBUTE]'
SET_ATTRIBUTE_TAG = b'[SET_ATTRIBUTE]'

EXCEPTION_TAG = b'[EXCEPTION]'
ATTRIBUTE_EXCEPTION_TAG = b'[ATTRIBUTE_EXCEPTION]'
SERIALIZE_EXCEPTION_TAG = b'[SERIALIZE_EXCEPTION]'
DESERIALIZE_EXCEPTION_TAG = b'[DESERIALIZE_EXCEPTION]'

NORMAL_TAG = b'[NORMAL]'

# interval of heartbeat mechanism in the unit of second
HEARTBEAT_INTERVAL_S = 10
HEARTBEAT_TIMEOUT_S = 10
HEARTBEAT_RCVTIMEO_S = HEARTBEAT_INTERVAL_S + HEARTBEAT_TIMEOUT_S * 2
