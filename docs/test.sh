#!/bin/bash
cd "$(dirname "$0")"
source ~/.bashrc
export PATH="/root/miniconda3/bin:$PATH"
source deactivate
source activate docs
pip install /work/
make html
