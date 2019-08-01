#!/bin/bash
xparl start --port 1234
xparl connect --address localhost:1234 --cpu_num 5
python train.py
