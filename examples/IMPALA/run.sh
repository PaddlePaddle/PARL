xparl start --port 1234
xparl connect --address localhost:1234  --cpu_num 32
sleep 3
python ./train.py
