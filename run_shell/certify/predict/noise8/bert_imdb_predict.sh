cd /home/zhangxinyu/code/fgws-main
#wait, 出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Predict

MODE=predict
DATASET=imdb
NOISE_TYPE=8
NOISE_PARAMETER=shuffle_len
MODEL_TYPE=bert
ATTACK=swap
AE_DATA=${ATTACK}_plain
LOG_PATH=/home/huangpeng/textRS/log/smooth/certify/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}/
LOG_FILE=${ATTACK}_test.log
GPU=3

mkdir -p ${LOG_PATH}

PARAMETER=64

CUDA_VISIBLE_DEVICES=$GPU python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ > ${LOG_PATH}/${LOG_FILE}


wait

PARAMETER=128

CUDA_VISIBLE_DEVICES=$GPU python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ > ${LOG_PATH}/${LOG_FILE}


wait

PARAMETER=256

CUDA_VISIBLE_DEVICES=$GPU python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_sh-len=${PARAMETER}/best_model/ > ${LOG_PATH}/${LOG_FILE}



