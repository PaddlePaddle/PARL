# use which GPU
export CUDA_VISIBLE_DEVICES=0

python evaluate.py --actor_num 160 \
           --difficulty 3 \
           --vel_penalty_coeff 3.0 \
           --penalty_coeff 2.0 \
           --saved_models_dir ./output/difficulty3/model_every_100_episodes \
           --evaluate_times 300
