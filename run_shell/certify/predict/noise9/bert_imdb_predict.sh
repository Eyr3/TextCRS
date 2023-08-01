cd /home/zhangxinyu/code/fgws-main
#wait, 出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Predict

MODE=predict
DATASET=imdb
NOISE_TYPE=9
MODEL_TYPE=bert
ATTACK=textfooler
LOG_PATH=/home/huangpeng/textRS/log/smooth/certify/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}/
LOG_FILE=${ATTACK}_test.log
GPU=0

mkdir -p ${LOG_PATH}

PARAMETER=50

CUDA_VISIBLE_DEVICES=$GPU python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -syn_size $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $ATTACK -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/ > ${LOG_PATH}/${LOG_FILE}


wait

PARAMETER=100

CUDA_VISIBLE_DEVICES=$GPU python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -syn_size $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $ATTACK -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/ > ${LOG_PATH}/${LOG_FILE}


wait

PARAMETER=250

CUDA_VISIBLE_DEVICES=$GPU python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -syn_size $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $ATTACK -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/ > ${LOG_PATH}/${LOG_FILE}



