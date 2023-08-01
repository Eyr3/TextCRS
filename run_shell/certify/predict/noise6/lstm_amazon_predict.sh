cd /home/zhangxinyu/code/fgws-main
#wait, 出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Predict

DATASET=amazon
NOISE_TYPE=6
NOISE_PARAMETER=syn_size
MODEL_TYPE=lstm
ATTACK=textfooler    #delete #insert #swap #textfooler
TEST_MODEL=checkpoint-epoch-30  #checkpoint-epoch-10, best_model
GPU=7

MODE=predict
AE_DATA=${ATTACK}
# AE_DATA=${ATTACK}_file
# AE_DATA=${ATTACK}_plain
# LOG_PATH=/home/huangpeng/textRS/log/smooth/certify/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}/
# LOG_FILE=${ATTACK}_test.log
# mkdir -p ${LOG_PATH}

PARAMETER=50

# benign accuracy: ae_data=None
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/ >/dev/null 2>&1 &
# wait
# PARAMETER=100
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/ >/dev/null 2>&1 &
# wait
# PARAMETER=250
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/ >/dev/null 2>&1 &


# predict robust accuracy with ae_data
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/ >/dev/null 2>&1 &

# DATASET=amazon
# PARAMETER=100
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &

# DATASET=imdb
# # PARAMETER=250
# CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/ >/dev/null 2>&1 &

