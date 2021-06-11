#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import time
import parl


@parl.remote_class
class Actor(object):
    def add_one(self, value):
        value += 1
        return value


def train():
    # reset_job_test.py will execute simulate_client.py, these two files must use the same port
    parl.connect('localhost:1337')  # can not use get_free_tcp_port()
    actor = Actor()
    actor.add_one(1)
    time.sleep(100000)


if __name__ == '__main__':
    train()
