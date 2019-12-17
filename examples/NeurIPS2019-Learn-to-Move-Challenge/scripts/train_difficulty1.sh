echo `which python` 
if [ $# != 1 ]; then
    echo "Usage: sh train_difficulty1.sh [RESTORE_MODEL_PATH]" 
    exit 0
fi

# use which GPU
export CUDA_VISIBLE_DEVICES=0

python train.py --actor_num 300 \
           --difficulty 1 \
           --penalty_coeff 3.0 \
           --logdir ./output/difficulty1 \
           --restore_model_path $1
