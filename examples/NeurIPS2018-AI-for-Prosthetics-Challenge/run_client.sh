#!/bin/bash
cd "$(dirname "$0")"

source ~/.bashrc


unset http_proxy
unset https_proxy

host=`hostname -i`

for index in {1..12};
do
    python3 simulator_client.py --port 9000 --ip $host --reward_type Round2 \
        --act_penalty_lowerbound 0.75 >> log1 &
done
wait
