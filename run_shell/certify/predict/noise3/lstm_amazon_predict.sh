#!/bin/bash
cd /home/zhangxinyu/code/fgws-main
#出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Predict

NOISE_TYPE=3
NOISE_PARAMETER=noise_sd
MODEL_TYPE=newlstm
ATTACK=insert
GPU=3
# NAME=VSD_loss
CHANNEL=1  # 2  4;  1  ; 0

MODE=predict
AE_DATA=${ATTACK}_file
# AE_DATA=${ATTACK}
TEST_MODEL=checkpoint-epoch-50  #best_model, checkpoint-epoch-50

PARAMETERS=(0.1)  # 0.1  0.5  1.0; 0.1  0.2  0.3
VSD_LOSS=(0)  # 0  5  10

# for ((i=0; i<${#PARAMETERS[@]}; i++))
# do
for ((j=0; j<${#VSD_LOSS[@]}; j++))
do
    PARAMETER=${PARAMETERS[j]}
    VSD=${VSD_LOSS[j]}

    echo "paramter=${PARAMETERS}"
    if [ $CHANNEL -ne 0 ]
    then
        # DATASET=imdb
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        # wait
        # DATASET=amazon
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        # wait
        TEST_MODEL=best_model
        DATASET=agnews
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
         
        # predict with clean dataset
        CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        wait
        PARAMETER=0.2
        CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        wait
        PARAMETER=0.3
        CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &

    else
        echo echo "CHANNEL=0" 
        # DATASET=imdb
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        # # wait
        DATASET=amazon
        CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        # wait
        # DATASET=agnews
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &

    fi    
done
# done

# VSD > 0
    #with alter _mu and no mu
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -VSD $VSD_LOSS -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/alterVSD${VSD_LOSS}_muchannel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/best_model/ >/dev/null 2>&1 &
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -VSD $VSD_LOSS -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/alterVSD${VSD_LOSS}_channel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/best_model/ >/dev/null 2>&1 &
    #no alter _mu and no mu
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -VSD $VSD_LOSS -dynamic_mu 1 -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/VSD${VSD_LOSS}.0_muchannel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/best_model/ >/dev/null 2>&1 &
        # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -VSD $VSD_LOSS -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/VSD${VSD_LOSS}.0_channel_${CHANNEL}noise${NOISE_TYPE}_g-n=${PARAMETER}/best_model/ >/dev/null 2>&1 &
