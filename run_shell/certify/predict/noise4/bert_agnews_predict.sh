#!/bin/bash
cd /home/zhangxinyu/code/fgws-main
#wait, 出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Predict

NOISE_TYPE=4
NOISE_PARAMETER=beta
MODEL_TYPE=bert
ATTACK=delete
GPU=4

MODE=predict
AE_DATA=${ATTACK}_file

PARAMETERS=(0.7)  # 0.3 0.5 0.7

for ((i=0; i<${#PARAMETERS[@]}; i++))
do
    PARAMETER=${PARAMETERS[i]}
    echo "Start beta=${PARAMETER}"
    # DATASET=agnews
    # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_beta=${PARAMETER}_sh-len=50/best_model/ >/dev/null 2>&1 &

    DATASET=amazon
    CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_beta=${PARAMETER}_sh-len=128/best_model/ >/dev/null 2>&1 &

    # DATASET=imdb
    # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_beta=${PARAMETER}_sh-len=256/best_model/ >/dev/null 2>&1 &
    
    # DATASET=agnews
    # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_beta=${PARAMETER}_sh-len=50/best_model/ >/dev/null 2>&1 &
    # wait
    # DATASET=amazon
    # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_beta=${PARAMETER}_sh-len=128/best_model/ >/dev/null 2>&1 &
    # wait
    # DATASET=imdb
    # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_beta=${PARAMETER}_sh-len=256/best_model/ >/dev/null 2>&1 &

done