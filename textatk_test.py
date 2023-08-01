import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
import argparse
from helper import set_batch_size, log_model_state_dict, set_lr, set_param
from helper import load_data_textatk, set_param, set_batch_size
from textattacknew import TrainingArgs, Trainer
from textattacknew.models.helpers import LSTMForClassification, NEWLSTMForClassification
from textattacknew.shared import logger
from textattacknew.models.wrappers import PyTorchModelWrapper


argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
argparser.add_argument("-model_type", type=str, default='lstm')
argparser.add_argument("-dataset", type=str, default='amazon')
argparser.add_argument("-model_path", type=str, default=None)
argparser.add_argument("-lr", type=float, default=1e-3)
args = argparser.parse_args()
model_type = args.model_type
dataset = args.dataset
lr = args.lr
glove_size = args.glove_size
output = '/data/xinyu/results/fgws/models/{}/{}/textatk_test/'.format(model_type, dataset)
max_len, bert_max_len, num_classes, val_split_size = set_param(dataset)
use_BERT = model_type == "bert" or model_type == "bert-G" or model_type == "roberta" or model_type == "roberta-G"
batch_size_train, batch_size_val, batch_size_test = set_batch_size(use_BERT, dataset)
if args.model_path is None:
    model_path = '/data/xinyu/results/fgws/models/{}/{}/textatk_lr={}_g=42B/best_model/'.format(model_type, dataset, lr)
    #best_model, checkpoint-epoch-50
else:
    model_path = args.model_path

logger.info("Loading textattacknew model: LSTMForClassification")
model = LSTMForClassification(
    max_seq_length=max_len,
    num_labels=num_classes,
    emb_layer_trainable=False,
    model_path=model_path,
    glove_size=glove_size,
)
model.cuda()
model_wrapper = PyTorchModelWrapper(model, model.tokenizer)

test_dataset = load_data_textatk(model_type, dataset, 'test')

training_args = TrainingArgs(num_epochs=1, learning_rate=lr, load_best_model_at_end=True, log_to_tb=True,
                             per_device_train_batch_size=batch_size_train, per_device_eval_batch_size=batch_size_test,
                             checkpoint_interval_epochs=1, output_dir=output, tb_log_dir='{}/logs/'.format(output))

test = Trainer(model_wrapper, "classification", None, None, test_dataset, training_args)

test.evaluate()
