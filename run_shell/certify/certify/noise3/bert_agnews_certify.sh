#!/bin/bash
cd /home/zhangxinyu/code/fgws-main
#wait, 出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Certify

MODE=certify
DATASET=agnews
NOISE_TYPE=3
NOISE_PARAMETER=noise_sd
MODEL_TYPE=bert
ATTACK=insert
GPU=0

PARAMETERS=(0.3)

for ((i=0; i<${#PARAMETERS[@]}; i++))
do 
    PARAMETER=${PARAMETERS[i]}
    
    CUDA_VISIBLE_DEVICES=$GPU python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_g-n=${PARAMETER}/best_model/ 

done

