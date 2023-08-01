cd /home/zhangxinyu/code/fgws-main
#wait, 出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Predict

MODE=test
NOISE_TYPE=6
NOISE_PARAMETER=syn_size
MODEL_TYPE=bert
GPU=5

ATTACK=swap  #delete #insert swap #textfooler
# AE_DATA=${ATTACK}
# AE_DATA=${ATTACK}_file
AE_DATA=${ATTACK}_plain
# LOG_PATH=/home/huangpeng/textRS/log/smooth/certify/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}/
# LOG_FILE=${ATTACK}_test.log
# mkdir -p ${LOG_PATH}
TEST_MODEL=best_model  #checkpoint-epoch-1  #best_model

# predict robust accuracy with ae_data

PARAMETER=50
DATASET=agnews
CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/ >/dev/null 2>&1 &

DATASET=amazon
CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/ >/dev/null 2>&1 &

DATASET=imdb
CUDA_VISIBLE_DEVICES=$GPU nohup python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/${TEST_MODEL}/ >/dev/null 2>&1 &

