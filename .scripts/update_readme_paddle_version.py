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

import os
import re


def update(fname, ver):
    with open(fname, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'paddlepaddle>=' in line:
            lines[i] = re.sub("paddlepaddle>=[\d+\.]+",
                              "paddlepaddle>={}".format(ver), line)

    with open(fname, 'w') as f:
        for line in lines:
            f.write(line)


if __name__ == '__main__':
    new_version = '1.6.1'

    readme_files = ['../README.md', '../README.cn.md']

    exclude_examples = [
        'NeurIPS2019-Learn-to-Move-Challenge',
        'NeurIPS2018-AI-for-Prosthetics-Challenge', 'LiftSim_baseline',
        'EagerMode'
    ]
    for example in os.listdir('../examples/'):
        if example not in exclude_examples:
            readme_files.append(
                os.path.join('../examples', example, 'README.md'))

    for example in os.listdir('../examples/EagerMode/'):
        readme_files.append(
            os.path.join('../examples/EagerMode', example, 'README.md'))

    print(readme_files)
    for fname in readme_files:
        update(fname, new_version)
