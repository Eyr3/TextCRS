#!/bin/bash

cd /home/zhangxinyu/code/fgws-main
#wait, 出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Certify

MODE=certify
DATASET=imdb
NOISE_TYPE=5
NOISE_PARAMETER=syn_size
MODEL_TYPE=lstm
ATTACK=textfooler
LOG_DIR=/home/huangpeng/textRS/fgws-main/huangpeng/certify/noise5/log
mkdir -p $LOG_DIR
GPU=1

PARAMETERS=(50 100 250)
for ((i=0; i<${#PARAMETERS[@]}; i++))
do 
    PARAMETER=${PARAMETERS[i]}
    
    CUDA_VISIBLE_DEVICES=$GPU python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/ > ${LOG_DIR}/${DATASET}_${MODEL_TYPE}_${PARAMETER}.log &
    wait

    echo "noise ${NOISE_TYPE} ${MODE} ${DATASET} ${MODEL_TYPE} ${PARAMETER} done"

done

