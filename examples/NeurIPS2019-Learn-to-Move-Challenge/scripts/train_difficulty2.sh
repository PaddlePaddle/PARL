if [ $# != 1 ]; then
    echo "Usage: sh train_difficulty2.sh [RESTORE_MODEL_PATH]" 
    exit 0
fi

# use which GPU
export CUDA_VISIBLE_DEVICES=0

python train.py --actor_num 300 \
           --difficulty 2 \
           --penalty_coeff 5.0 \
           --logdir ./output/difficulty2 \
           --restore_model_path $1
