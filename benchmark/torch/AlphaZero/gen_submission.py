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

import sys
import base64
import inspect
import os

assert len(sys.argv) == 2, "please specify model path."
model_path = sys.argv[1]

with open(model_path, 'rb') as f:
    raw_bytes = f.read()
    encoded_weights = base64.encodebytes(raw_bytes)

# encode weights of model to byte string
submission_file = """
import base64
decoded = base64.b64decode({})

""".format(encoded_weights)

# insert code snippet of loading weights
with open('submission_template.py', 'r') as f:
    submission_file += ''.join(f.readlines())

# generate final submission file
with open('submission.py', 'w') as f:
    f.write(submission_file)
