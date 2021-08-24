#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import base64
import pickle
import paddle

if __name__ == '__main__':

    model = paddle.load('./model/latest_ship_model.pth')
    actor = model['actor']

    for name, param in actor.items():
        actor[name] = param.numpy()

    model_byte = base64.b64encode(pickle.dumps(actor))
    with open('./model/actor.txt', 'wb') as f:
        f.write(model_byte)
