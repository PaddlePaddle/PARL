if [ $# != 1 ]; then
    echo "Usage: sh train_difficulty3.sh [RESTORE_MODEL_PATH]" 
    exit 0
fi

# use which GPU
export CUDA_VISIBLE_DEVICES=0

python train.py --actor_num 300 \
           --difficulty 3 \
           --vel_penalty_coeff 3.0 \
           --penalty_coeff 2.0 \
           --rpm_size 6e6 \
           --train_times 250 \
           --logdir ./output/difficulty3 \
           --restore_model_path $1
