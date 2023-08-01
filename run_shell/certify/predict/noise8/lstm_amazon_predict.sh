#!/bin/bash
cd /home/zhangxinyu/code/fgws-main

MODE=predict

NOISE_TYPE=8
NOISE_PARAMETER=shuffle_len
MODEL_TYPE=lstm
ATTACK=swap
AE_DATA=${ATTACK}_file
LOG_DIR=/home/huangpeng/textRS/fgws-main/huangpeng/noise8/log
mkdir -p $LOG_DIR

# GPU=4
# PARAMETERS=(64 128)

# for ((i=0; i<${#PARAMETERS[@]}; i++))
# do 
#     PARAMETER=${PARAMETERS[i]}

#     CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ > ${LOG_DIR}/${DATASET}_${MODEL_TYPE}_${PARAMETER}.log & 

#     echo "Start $MODE  Noise${NOISE_TYPE} $DATASET $MODEL_TYPE PARAMETER=$PARAMETER"
# done

GPU=3
DATASET=imdb
# PARAMETERS=(256)

# for ((i=0; i<${#PARAMETERS[@]}; i++))
# do 
#     PARAMETER=${PARAMETERS[i]}

#     CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ >/dev/null 2>&1 &

#     echo "Start $MODE  Noise${NOISE_TYPE} $DATASET $MODEL_TYPE PARAMETER=$PARAMETER"
# done

# PARAMETER=32
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ >/dev/null 2>&1 &
PARAMETER=256
CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ >/dev/null 2>&1 &
PARAMETER=128
CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ >/dev/null 2>&1 &
