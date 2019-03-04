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


# A dev image based on paddle production image

FROM paddlepaddle/paddle:1.1.0-gpu-cuda9.0-cudnn7

Run apt-get update
RUN apt-get install -y cmake
# Prepare packages for Python
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev

# Install python3.6
RUN wget -q https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tgz && \
    tar -xzf Python-3.6.0.tgz && cd Python-3.6.0 && \
    CFLAGS="-Wformat" ./configure --prefix=/usr/local/ --enable-shared && \
    make -j8  && make altinstall && \
    cp libpython3.6m.so.1.0 /usr/lib/ && cp libpython3.6m.so.1.0 /usr/local/lib/

COPY ./requirements.txt /root/

# Requirements for python2
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /root/requirements.txt

# Requirements for python3
RUN pip3.6 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /root/requirements.txt
RUN pip3.6 install -i https://pypi.tuna.tsinghua.edu.cn/simple paddlepaddle-gpu==1.3.0.post97
