#!/bin/bash
cd /home/zhangxinyu/code/fgws-main
#wait, 出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Certify

MODE=certify
NOISE_TYPE=3
NOISE_PARAMETER=noise_sd
MODEL_TYPE=newcnn
TEST_MODEL=checkpoint-epoch-50  #best_model, checkpoint-epoch-50
CHANNEL=1  # 1 0
GPU=4

ATTACK=insert  #insert, bae_i, delete, textfooler, swap
AE_DATA=${ATTACK}_file
# AE_DATA=${ATTACK}

# PARAMETERS=(0.1 0.2 0.3)  #
# VSD=0

# for ((i=0; i<${#PARAMETERS[@]}; i++))
# do 
#     PARAMETER=${PARAMETERS[i]}

#     if [ $CHANNEL -ne 0 ]
#     then  # CHANNEL=1 with mu
#         DATASET=agnews
#         # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
#         # DATASET=amazon
#         # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
#         # DATASET=imdb
#         # TEST_MODEL=checkpoint-epoch-50
#         # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &

#         # PARAMETER=0.2
#         # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
#         # PARAMETER=0.3
#         # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &

#     else  #without mu
#         echo "CHANNEL=0"
#         # DATASET=agnews
#         # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
#         # DATASET=amazon
#         # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
#         # DATASET=imdb
#         # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
        
#         # PARAMETER=0.2
#         # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
#         # PARAMETER=0.3
#         # CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &

#     fi
# done

DATASET=amazon
# PARAMETER=0.1
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -N 100000 -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
# PARAMETER=0.2
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -N 100000 -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
PARAMETER=0.3
CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -N 100000 -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/from_0.1muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &


# ======================== certify with ae_data ========================
# DATASET=imdb
# PARAMETER=0.1
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -N 20000 -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
# PARAMETER=0.2
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -N 20000 -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
# # wait
# PARAMETER=0.3
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -N 20000 -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/from_0.1muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &


# DATASET=amazon
# PARAMETER=0.2
# TEST_MODEL=checkpoint-epoch-50
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
# wait
# PARAMETER=0.3
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &


# DATASET=imdb
# TEST_MODEL=checkpoint-epoch-50
# PARAMETER=0.2
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
# wait
# PARAMETER=0.3
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -dynamic_mu 1 -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/muchannel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &



# ======================== channel=0 certify with ae_data ========================
# echo "CHANNEL=0"

# DATASET=agnews
# PARAMETER=0.1
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
# wait
# PARAMETER=0.2
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
# wait
# PARAMETER=0.3
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &


DATASET=amazon
TEST_MODEL=checkpoint-epoch-50
# PARAMETER=0.1
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
# wait
# PARAMETER=0.2
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
# wait
# PARAMETER=0.3
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &


# DATASET=imdb
# TEST_MODEL=checkpoint-epoch-50
# PARAMETER=0.1
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
# wait
# PARAMETER=0.2
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
# wait
# PARAMETER=0.3
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -channel_rate $CHANNEL -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/channel_${CHANNEL}.0noise${NOISE_TYPE}_g-n=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &
