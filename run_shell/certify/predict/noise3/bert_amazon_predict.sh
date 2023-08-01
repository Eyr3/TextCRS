#!/bin/bash
cd /home/zhangxinyu/code/fgws-main
#出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Predict

MODE=predict
NOISE_TYPE=3
NOISE_PARAMETER=noise_sd
MODEL_TYPE=newbert
ATTACK=insert  #insert, bae_i, delete, textfooler, swap
AE_DATA=${ATTACK}_file
# AE_DATA=${ATTACK}
# AE_DATA=${ATTACK}_plain 
TEST_MODEL=best_model  #checkpoint-epoch-10, best_model

GPU=6
CHANNEL=0.0  # 4.0 0.0

PARAMETERS=(1.5)  # 0.5 1.0 1.5
VSD=0 # 0 5

for ((i=0; i<${#PARAMETERS[@]}; i++))
do
    PARAMETER=${PARAMETERS[i]}
    
    if [ $VSD -ne 0 ]
    then
    # with mu
        echo "VSD=${VSD}" 
        # DATASET=agnews
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -VSD_loss $VSD -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/VSD${VSD}.0_muchannel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        DATASET=amazon
        CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -VSD_loss $VSD -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/VSD${VSD}.0_muchannel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        # DATASET=imdb
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -VSD_loss $VSD -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/VSD${VSD}.0_muchannel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        
        # without mu
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -VSD_loss $VSD -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/VSD${VSD}.0_channel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &

    else
        echo "VSD=0"          
        DATASET=agnews
        CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        # DATASET=amazon
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        # DATASET=imdb
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        
        # with mu
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/best_model/ >/dev/null 2>&1 &
    
        # predict with clean dataset:
        # DATASET=agnews
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        # PARAMETER=1.0
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        # PARAMETER=1.5
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &

    fi
    # wait   # ${NAME}
done
# done