#!/bin/bash
xparl start --port 1234
xparl connect --address localhost:1234 --cpu_num 32 
sleep 2
python train.py

