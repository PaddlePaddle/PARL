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
import os
import re
from setuptools import setup, find_packages


def _find_packages(prefix=''):
    packages = []
    path = '.'
    prefix = prefix
    for root, _, files in os.walk(path):
        if '__init__.py' in files:
            packages.append(re.sub('^[^A-z0-9_]', '', root.replace('/', '.')))
    return packages


setup(
    name='parl',
    version=1.1,
    packages=_find_packages(),
    package_data={'': ['*.so']},
    install_requires=[
        "termcolor>=1.1.0",
        "pyzmq>=17.1.2",
        "pyarrow>=0.12.0",
    ],
)
