# use which GPU
export CUDA_VISIBLE_DEVICES=0


python evaluate.py --actor_num 160 \
           --difficulty 2 \
           --penalty_coeff 5.0 \
           --saved_models_dir ./output/difficulty2/model_every_100_episodes \
           --evaluate_times 300
