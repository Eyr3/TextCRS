cd /home/huangpeng/textRS/fgws-main
#wait, 出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# Peng Huang
## Predict

DATASET=agnews
NOISE_TYPE=5
NOISE_PARAMETER=syn_size
MODEL_TYPE=bert
ATTACK=textfooler
GPU=2

MODE=predict
AE_DATA=${ATTACK}

PARAMETER=50

CUDA_VISIBLE_DEVICES=$GPU python -u textatk_train.py -mode $MODE -model_type $MODEL_TYPE -dataset $DATASET -if_addnoise $NOISE_TYPE -$NOISE_PARAMETER $PARAMETER -sigma $PARAMETER -atk $ATTACK -ae_data $AE_DATA -model_path /data/xinyu/results/fgws/models/${MODEL_TYPE}/${DATASET}/noise${NOISE_TYPE}_k=${PARAMETER}/best_model/