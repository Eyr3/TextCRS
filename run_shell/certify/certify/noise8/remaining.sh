#!/bin/bash
cd /home/zhangxinyu/code/fgws-main
#wait, 出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Certify

MODE=certify
DATASET=imdb
NOISE_TYPE=8
NOISE_PARAMETER=shuffle_len
MODEL_TYPE=bert
ATTACK=swap
LOG_DIR=/home/huangpeng/textRS/fgws-main/huangpeng/certify/noise8/log
mkdir -p $LOG_DIR
GPU=0

PARAMETERS=(128 256)
for ((i=0; i<${#PARAMETERS[@]}; i++))
do 
    PARAMETER=${PARAMETERS[i]}
    
    CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ > ${LOG_DIR}/${DATASET}_${MODEL_TYPE}_${PARAMETER}.log & 

done

DATASET=amazon
PARAMETERS=(128)

for ((i=0; i<${#PARAMETERS[@]}; i++))
do 
    PARAMETER=${PARAMETERS[i]}
    
    CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ > ${LOG_DIR}/${DATASET}_${MODEL_TYPE}_${PARAMETER}.log 

done

