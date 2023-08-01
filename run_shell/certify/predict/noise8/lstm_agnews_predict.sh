#!/bin/bash
cd /home/zhangxinyu/code/fgws-main

MODE=predict
DATASET=agnews
NOISE_TYPE=8
NOISE_PARAMETER=shuffle_len
MODEL_TYPE=lstm
ATTACK=swap
AE_DATA=${ATTACK}_file
LOG_DIR=/home/huangpeng/textRS/fgws-main/huangpeng/noise8/log
mkdir -p $LOG_DIR

GPU=4
PARAMETERS=(32 64)

for ((i=0; i<${#PARAMETERS[@]}; i++))
do 
    PARAMETER=${PARAMETERS[i]}

    CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ > ${LOG_DIR}/${DATASET}_${MODEL_TYPE}_${PARAMETER}.log & 

    echo "Start $MODE  Noise${NOISE_TYPE} $DATASET $MODEL_TYPE PARAMETER=$PARAMETER"
done

GPU=5
PARAMETERS=(128)

for ((i=0; i<${#PARAMETERS[@]}; i++))
do 
    PARAMETER=${PARAMETERS[i]}

    CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ > ${LOG_DIR}/${DATASET}_${MODEL_TYPE}_${PARAMETER}.log & 

    echo "Start $MODE  Noise${NOISE_TYPE} $DATASET $MODEL_TYPE PARAMETER=$PARAMETER"
done

