#!/bin/bash

export CPU_NUM=1

actor_num=48

for i in $(seq 1 $actor_num); do
    python actor.py &
done;
wait
