#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

for index in {1..32}; do
    python actor.py &
done;
wait
