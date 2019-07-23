#!/bin/bash
cd "$(dirname "$0")"
source ~/.bashrc
export PATH="/root/miniconda3/bin:$PATH"
source activate docs
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple /work/
make html
