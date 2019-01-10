## Curriculum learning

### Run Fastest
```
# server
python .simulator_server.py \
           --port [PORT] \
           --ensemble_num 1 \

# client
python simulator_client.py \
           --port [PORT] \
           --ip [IP] \
           --reward_type RunFastest \
```

### Fixed Target Speed
#### target speed 3.0 m/s
```
# server
python .simulator_server.py \
           --port [PORT] \
           --ensemble_num 1 \
           --restore_model_path [RunFastest model] \
           --warm_start_batchs 1000 \

# client
python simulator_client.py \
           --port [PORT] \
           --ip [IP] \
           --reward_type FixedTargetSpeed \
           --target_v 3.0 \
           --act_penalty_lowerbound 1.5 \
```

#### target speed 2.0 m/s
```
# server
python .simulator_server.py \
           --port [PORT] \
           --ensemble_num 1 \
           --restore_model_path [FixedTargetSpeed 3.0m/s model] \
           --warm_start_batchs 1000 \

# client
python simulator_client.py \
           --port [PORT] \
           --ip [IP] \
           --reward_type FixedTargetSpeed \
           --target_v 2.0 \
           --act_penalty_lowerbound 0.75 \
```

#### target speed 1.25 m/s
```
# server
python .simulator_server.py \
           --port [PORT] \
           --ensemble_num 1 \
           --restore_model_path [FixedTargetSpeed 2.0m/s model] \
           --warm_start_batchs 1000 \

# client
python simulator_client.py \
           --port [PORT] \
           --ip [IP] \
           --reward_type FixedTargetSpeed \
           --target_v 1.25 \
           --act_penalty_lowerbound 0.6 \
```

## one-head model to multi-head model script

## Round2 
> Pretrained multi-head 1.25m/s model: [link]()

```
# server
python .simulator_server.py \
           --port [PORT] \
           --ensemble_num 12 \
           --restore_model_path [FixedTargetSpeed 1.25m/s multi-head model] \
           --warm_start_batchs 1000 \

# client
python simulator_client.py \
           --port [PORT] \
           --ip [IP] \
           --reward_type Round2 \
           --act_penalty_lowerbound 0.75 \
           --act_penalty_coeff 7.0 \
           --vel_penalty_coeff 20.0 \
           --discrete_data \
           --stage 3 \
```
