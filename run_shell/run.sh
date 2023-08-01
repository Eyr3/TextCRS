# conda activate torch19
cd /home/zhangxinyu/code/fgws-main
# excute: sh run_shell/run.sh
#wait, 出现包没有的问题，先运行一遍textattack attack，再将/home/zhangxinyu/.cache/textattack/word_embeddings 复制到textattacknew中

# CUDA_VISIBLE_DEVICES=4 nohup python -u main.py -dataset amazon -model_type bert -name new -visible_devices 4 > /home/zhangxinyu/code/fgws-main/log/amazon/newbert.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python -u main.py -dataset agnews -model_type bert -name newbert_ml=128 > /home/zhangxinyu/code/fgws-main/log/agnews/newbert_ml=128.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python -u main.py -dataset imdb -model_type bert -name new -visible_devices 5 > /home/zhangxinyu/code/fgws-main/log/imdb/newbert.log 2>&1 &
#-dataset agnews -model_type bert -name test -if_addnoise 3 -noise_sd 0 -freeze_layer 0

# test with ae: -mode test -dataset agnews -model_type bert -ae_data test
# CUDA_VISIBLE_DEVICES=4 python -u textatk_train.py -mode test -dataset amazon -model_type newcnn -model_path /data/xinyu/results/fgws/models/newcnn/amazon/muchannel_1.0noise3_g-n=0.3/checkpoint-epoch-50
# # # wait
# CUDA_VISIBLE_DEVICES=4 python -u textatk_train.py -mode test -dataset amazon -model_type newcnn -channel_rate 1 -dynamic_mu True -model_path /data/xinyu/results/fgws/models/newcnn/amazon/muchannel_1.0noise3_g-n=0.3/checkpoint-epoch-50
# # # test on ae_data
# CUDA_VISIBLE_DEVICES=5 python -u textatk_train.py -mode test -dataset amazon -model_type newbert -channel_rate 0.0 -atk insert -ae_data insert_file -model_path /data/xinyu/results/fgws/models/newbert/amazon/channel_0.0noise3_g-n=1.0/best_model

# # CUDA_VISIBLE_DEVICES=5 nohup python -u textatk_train.py -mode predict -model_type bert -channel_rate 0 -dataset agnews -if_addnoise 3 -noise_sd 0 -sigma 0 -atk insert -ae_data insert_file_0.8 -model_path /data/xinyu/results/fgws/models/bert/agnews/noise3_g-n=0.0/best_model >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python -u textatk_train.py -mode predict -model_type newbert -channel_rate 2.0 -dataset amazon -if_addnoise 3 -noise_sd 0.5 -sigma 0.5 -atk insert -ae_data insert_file -model_path /data/xinyu/results/fgws/models/newbert/amazon/VSD5.0_channel_2.0noise3_g-n=0.5/best_model/ >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode certify -dataset agnews -model_type cnn -if_addnoise 3 -noise_sd 0.1 -sigma 0.5 >/dev/null 2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode certify -dataset agnews -model_type lstm -if_addnoise 5 -noise_sd 0.1 -sigma 0.5 -model_path /data/xinyu/results/fgws/models/lstm/agnews/best/best_model >/dev/null 2>&1 &

## imdb
# CUDA_VISIBLE_DEVICES=5 nohup python -u textatk_train.py -mode train -dataset agnews -model_type bert -if_addnoise 9 -syn_size 50 > /home/zhangxinyu/code/fgws-main/log/noise/agnews/bert_9_k=50.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u textatk_train.py -mode train -dataset agnews -model_type lstm -if_addnoise 1 -syn_size 50 -lr 1e-4 -start_epoch 21 -model_path /data/xinyu/results/fgws/models/lstm/agnews/noise1_k=50/checkpoint-epoch-20/ > /home/zhangxinyu/code/fgws-main/log/noise/agnews/lstm_1_k=50_test.log 2>&1 &
# # wait
# CUDA_VISIBLE_DEVICES=2 nohup python -u textatk_train.py -mode train -dataset agnews -model_type newbert -if_addnoise 3 -noise_sd 0.5 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset imdb -model_type cnn -if_addnoise 4 -beta 0.3 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset imdb -model_type cnn -if_addnoise 4 -beta 0.5 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python -u textatk_train.py -mode train -dataset imdb -model_type cnn -if_addnoise 4 -beta 0.7 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset amazon -model_type lstm -if_addnoise 4 -beta 0.9 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u textatk_train.py -mode train -dataset imdb -model_type bert -if_addnoise 4 -beta 0.9 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset imdb -model_type lstm -if_addnoise 4 -beta 0.9 >/dev/null 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset agnews -model_type bert -if_addnoise 4 -beta 0.5 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset agnews -model_type bert -if_addnoise 4 -beta 0.25 >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset imdb -model_type cnn -if_addnoise 8 -shuffle_len 256 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset imdb -model_type cnn -if_addnoise 8 -shuffle_len 64 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset imdb -model_type cnn -if_addnoise 8 -shuffle_len 128 >/dev/null 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=0 nohup python -u textatk_train.py -mode train -dataset imdb -model_type bert -if_addnoise 1 -syn_size 250 >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset agnews -model_type bert -if_addnoise 4 -beta 0.3 >/dev/null 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset amazon -model_type bert -if_addnoise 4 -beta 0.3 >/dev/null 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset imdb -model_type bert -if_addnoise 4 -beta 0.3 >/dev/null 2>&1 &
# # wait
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset amazon -model_type cnn -if_addnoise 3 -noise_sd 0.2 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python -u textatk_train.py -mode train -dataset amazon -model_type cnn -if_addnoise 3 -noise_sd 0.3 >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset amazon -model_type lstm -if_addnoise 6 -syn_size 50 > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=5 nohup python -u textatk_train.py -mode train -dataset amazon -model_type lstm -if_addnoise 6 -syn_size 50 > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset imdb -model_type cnn -if_addnoise 5 -syn_size 50 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset imdb -model_type cnn -if_addnoise 5 -syn_size 100 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python -u textatk_train.py -mode train -dataset imdb -model_type cnn -if_addnoise 5 -syn_size 250 >/dev/null 2>&1 &


# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset amazon -model_type bert -if_addnoise 6 -syn_size 50 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset amazon -model_type bert -if_addnoise 6 -syn_size 100 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset amazon -model_type bert -if_addnoise 6 -syn_size 250 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset imdb -model_type lstm -if_addnoise 5 -syn_size 250 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset agnews -model_type lstm -if_addnoise 6 -syn_size 50 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python -u textatk_train.py -mode train -dataset agnews -model_type bert -if_addnoise 6 -syn_size 250 > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u textatk_train.py -mode train -dataset imdb -model_type newlstm -channel_rate 0.5 -if_addnoise 3 -noise_sd 0.2 -name dynamic >/dev/null 2>&1 &
# wait

# CUDA_VISIBLE_DEVICES=5 nohup python -u textatk_train.py -mode train -dataset imdb -model_type newlstm -channel_rate 2 -if_addnoise 3 -noise_sd 1.0 -name dynamic >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset imdb -model_type newlstm -channel_rate 0.5 -if_addnoise 3 -noise_sd 0.2 -name channel_0.5 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u textatk_train.py -mode train -dataset amazon -model_type lstm -if_addnoise 3 -noise_sd 0.2 -name dynamic_mu >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset imdb -model_type lstm -if_addnoise 3 -noise_sd 0.2 -name channel >/dev/null 2>&1 &


### without enmhancement, noise3 (noise_sd) must use newlstm/newbert
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset agnews -model_type newbert -channel_rate 0 -if_addnoise 3 -noise_sd 2 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newbert -channel_rate 0 -if_addnoise 3 -noise_sd 2 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset imdb -model_type newbert -channel_rate 0 -if_addnoise 3 -noise_sd 2 >/dev/null 2>&1 &


#channel without kl-divergence and vsd=0
CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newcnn -channel_rate 1 -if_addnoise 3 -noise_sd 0.3 -dynamic_mu True -model_path /data/xinyu/results/fgws/models/newcnn/amazon/muchannel_1.0noise3_g-n=0.1/best_model >/dev/null -name from_0.1 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newcnn -channel_rate 1 -if_addnoise 3 -noise_sd 0.3 -dynamic_mu True >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset imdb -model_type newcnn -channel_rate 1 -if_addnoise 3 -noise_sd 0.3 -dynamic_mu True >/dev/null 2>&1 &
# without dynamic_mu
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newbert -channel_rate 0 -if_addnoise 3 -noise_sd 1.5 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newbert -channel_rate 0 -if_addnoise 3 -noise_sd 1.5 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset imdb -model_type newbert -channel_rate 0 -if_addnoise 3 -noise_sd 1.5 -model_path /data/xinyu/results/fgws/models/newbert/imdb/channel_0.0noise3_g-n=0.1/best_model >/dev/null 2>&1 & 

# -name alter

#channel with kl-divergence and vsd=5 / 10
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newbert -channel_rate 0.5 -if_addnoise 3 -noise_sd 0.1 -use_kl True -VSD_loss 5 -dynamic_mu True >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newbert -channel_rate 0.5 -if_addnoise 3 -noise_sd 0.2 -use_kl True -VSD_loss 5 -dynamic_mu True >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newbert -channel_rate 0.5 -if_addnoise 3 -noise_sd 0.3 -use_kl True -VSD_loss 5 -dynamic_mu True >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset imdb -model_type newbert -channel_rate 4 -if_addnoise 3 -noise_sd 0.5 -use_kl True -VSD_loss 5 -dynamic_mu True >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newbert -channel_rate 4 -if_addnoise 3 -noise_sd 1.5 -use_kl True -VSD_loss 5 -dynamic_mu True >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python -u textatk_train.py -mode train -dataset agnews -model_type newbert -channel_rate 4 -if_addnoise 3 -noise_sd 1.5 -use_kl True -VSD_loss 5 -dynamic_mu True >/dev/null 2>&1 &

# # without dynamic_mu
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newbert -channel_rate 0.5 -if_addnoise 3 -noise_sd 0.1 -use_kl True -VSD_loss 5  >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newbert -channel_rate 0.5 -if_addnoise 3 -noise_sd 0.2 -use_kl True -VSD_loss 5 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newbert -channel_rate 0.5 -if_addnoise 3 -noise_sd 0.3 -use_kl True -VSD_loss 5 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset imdb -model_type newbert -channel_rate 4 -if_addnoise 3 -noise_sd 0.1 -use_kl True -VSD_loss 5 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset imdb -model_type newbert -channel_rate 4 -if_addnoise 3 -noise_sd 0.5 -use_kl True -VSD_loss 5 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset imdb -model_type newbert -channel_rate 4 -if_addnoise 3 -noise_sd 1.0 -use_kl True -VSD_loss 5 >/dev/null 2>&1 &

# wait
# CUDA_VISIBLE_DEVICES=0 nohup python -u textatk_train.py -mode train -dataset amazon -model_type newbert -if_addnoise 3 -noise_sd 0.05 -name dynamic -channel 0 >/dev/null 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset imdb -model_type newbert -if_addnoise 3 -noise_sd 0.05 >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset imdb -model_type bert -if_addnoise 2 -syn_size 50 >/dev/null 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset imdb -model_type bert -if_addnoise 2 -syn_size 100 >/dev/null 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=1 nohup python -u textatk_train.py -mode train -dataset imdb -model_type bert -if_addnoise 2 -syn_size 250 >/dev/null 2>&1 &


## train clean vanilla
# CUDA_VISIBLE_DEVICES=4 nohup python -u textatk_train.py -mode train -dataset agnews -model_type lstm -name best >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset amazon -model_type cnn -name best >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_train.py -mode train -dataset imdb -model_type cnn -name best >/dev/null 2>&1 &


## lstm
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_attack.py -model_type cnn -dataset amazon -atk combine -num_examples 1000 -mode test > /data/xinyu/results/fgws/attacks/cnn/amazon/textatk_combine.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_attack.py -model_type cnn -dataset agnews -atk combine -num_examples 1000 -mode test > /data/xinyu/results/fgws/attacks/cnn/agnews/textatk_combine.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_attack.py -model_type cnn -dataset imdb -atk combine -num_examples 1000 -mode test > /data/xinyu/results/fgws/attacks/cnn/imdb/textatk_combine.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python -u textatk_attack.py -model_type cnn -dataset amazon -atk textfooler -num_examples 1000 -mode test > /data/xinyu/results/fgws/attacks/cnn/amazon/textatk_textfooler.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python -u textatk_attack.py -model_type cnn -dataset amazon -atk delete -num_examples 1000 -mode test > /data/xinyu/results/fgws/attacks/cnn/amazon/textatk_delete.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python -u textatk_attack.py -model_type cnn -dataset imdb -atk delete -num_examples 1000 -mode test > /data/xinyu/results/fgws/attacks/cnn/imdb/textatk_delete.log 2>&1 &
## bert
# --------! modify the log name !--------
# CUDA_VISIBLE_DEVICES=5 nohup python -u textatk_attack.py -model_type bert -dataset agnews -atk insert -num_examples 1000 -mode test > /data/xinyu/results/fgws/attacks/bert/agnews/textatk_insert.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 nohup python -u textatk_attack.py -model_type bert -dataset amazon -atk clare -num_examples 200 -mode test > /data/xinyu/results/fgws/attacks/bert/amazon/textatk_clare.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u textatk_attack.py -model_type bert -dataset imdb -atk bae_i -num_examples 200 -mode test > /data/xinyu/results/fgws/attacks/bert/imdb/textatk_bae_i.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u textatk_train.py -mode predict -model_type bert -dataset imdb -if_addnoise 1 -syn_size 50 -sigma 50 -atk textfooler -ae_data textfooler -model_path /data/xinyu/results/fgws/models/bert/imdb/noise1_k=50/best_model/ >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u textatk_train.py -mode predict -model_type bert -dataset agnews -if_addnoise 1 -syn_size 50 -sigma 50 -ae_data textfooler -model_path /data/xinyu/results/fgws/models/bert/agnews/noise9_k=50/best_model >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup textattack attack --recipe textfooler --model bert-base-uncased-imdb --num-successful-examples 1000 \
#    --checkpoint-dir /data/xinyu/results/fgws/attacks/checkpoint \
#    --log-to-csv /data/xinyu/results/fgws/attacks/textattack/default_bert_imdb_textfooler.csv \
#    --csv-coloring-style plain --shuffle --log-summary-to-json /data/xinyu/results/fgws/attacks/textattack/default_bert_imdb_textfooler.json \
# #    > /home/zhangxinyu/code/fgws-main/log/textattack/default_lstm_imdb_textfooler.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup textattack attack --recipe bae --model lstm-ag-news --num-successful-examples 1000 \
#    --checkpoint-dir /data/xinyu/results/fgws/attacks/checkpoint \
#    --log-to-csv /data/xinyu/results/fgws/attacks/textattack/default_lstm_agnews_bae.csv \
#    --csv-coloring-style plain --shuffle --log-summary-to-json /data/xinyu/results/fgws/attacks/textattack/default_lstm_agnews_bae.json \
#     >/data/xinyu/results/fgws/attacks/textattack/default_lstm_agnews_bae.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup textattack attack --recipe textfooler --model bert-base-uncased-imdb --num-successful-examples 1000 \
#    --checkpoint-dir /data/xinyu/results/fgws/attacks/checkpoint \
#    --log-to-csv /data/xinyu/results/fgws/attacks/textattack/default_bert_imdb_textfooler.csv \
#    --csv-coloring-style plain --shuffle --log-summary-to-json /data/xinyu/results/fgws/attacks/textattack/default_bert_imdb_textfooler.json \
#    > /home/zhangxinyu/code/fgws-main/log/textattack/default_lstm_imdb_textfooler.txt 2>&1 &

