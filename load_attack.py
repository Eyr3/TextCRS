import os
import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from copy import deepcopy
from config import Config
from logger import Logger
from data_module import DataModule
from models.cnn import CNN, CNN_for_textattack 
from models.lstm import LSTM
from models.bert_wrapper import BertWrapper
from utils import (
    prep_seq,
    pad,
    load_model,
    compute_accuracy,
    inference,
    shuffle_lists,
    list_join,
    copy_file,
    load_pkl,
)
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse 
from textattack.models.wrappers import ModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import PWWSRen2019, TextFoolerJin2019, GeneticAlgorithmAlzantot2018, BERTAttackLi2020, DeepWordBugGao2018, TextBuggerLi2018, PSOZang2020, BAEGarg2019, FasterGeneticAlgorithmJia2019
from textattack import AttackArgs
from textattack.datasets import Dataset
from textattack import Attacker
import textattack
import sys 
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, BertConfig 
from transformers import BertForSequenceClassification, AdamW
# from transformers import get_linear_schedule_with_warmup
from spacy.lang.en import English

try:
    import cPickle as pickle
except ImportError:
    import pickle

def load_state(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def load_for_textattack(config):
    # config = Config()
    # logger = Logger(config)
    # data_module = DataModule(config, logger)
    # config.vocab_size = len(data_module.vocab)
    config.vocab_size = 64786
    config.w2i = load_pkl("{}/{}".format("/public1014/zhub/fgws/data/models/cnn/imdb/data", "word_to_idx.pkl"))
    bert_wrapper = None

    # if config.mode == "train":
    #     train_writer = SummaryWriter(config.tb_log_train_path)
    #     val_writer = SummaryWriter(config.tb_log_val_path)
    #     criterion = nn.CrossEntropyLoss()
    #     best_epoch = (0, np.inf)
    #     optimizer, scheduler = None, None

    #     copy_file(config)

    #     if config.use_BERT:
    #         bert_wrapper = BertWrapper(config, logger)
    #         model = bert_wrapper.model

    #         optimizer = AdamW(
    #             model.parameters(),
    #             lr=config.learning_rate,
    #             eps=config.adam_eps,
    #             weight_decay=config.weight_decay,
    #         )

    #         total_steps = config.num_epoch * int(
    #             math.ceil(len(data_module.train_texts) / config.batch_size_train)
    #         )

    #         scheduler = get_linear_schedule_with_warmup(
    #             optimizer,
    #             num_warmup_steps=int(total_steps * config.warmup_percent),
    #             num_training_steps=total_steps,
    #         )
    #     else:
    #         if config.model_type == "cnn":
    #             model = CNN(
    #                 config, logger, pre_trained_embs=data_module.init_pretrained
    #             )
    #         else:
    #             model = LSTM(
    #                 config, logger, pre_trained_embs=data_module.init_pretrained
    #             )

    #         optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    #     if config.gpu:
    #         model.cuda()

    #     logger.log.info("Start training")

    #     for epoch in range(1, config.num_epoch + 1):
    #         run_epoch(epoch, scheduler=scheduler, bert_wrapper=bert_wrapper)

    #     logger.log.info("Finished training")

    #     model = load_model(
    #         "{}/checkpoints/e_best/model.pth".format(config.model_base_path),
    #         model,
    #         logger,
    #     )
    #     test_model(bert_wrapper=bert_wrapper)
    # elif config.mode == "test":
    # if config.mode == "test":
    if config.use_BERT:
        bert_wrapper = BertWrapper(config, logger)
        model = bert_wrapper.model
    elif config.model_type == "cnn":
        # model = CNN(config, logger)
        model = CNN_for_textattack(config)
    else:
        model = LSTM(config, logger)

    # model = load_model(config.load_model_path, model, logger)
    model = load_state("/public1014/zhub/fgws/"+config.load_model_path, model)

    if config.gpu:
        model.cuda()

        # test_model(bert_wrapper=bert_wrapper)
    # else:
    #     logger.log.info("Incorrect mode {}. Exit.".format(config.mode))
    #     exit()
    return model 

def inference(
    inputs,
    model,
    word_to_idx,
    config,
    bert_wrapper=None,
    tokenizer=None,
    val=False,
    single=False,
):
    softmax = nn.Softmax(dim=1)
    model.eval()

    if single:
        assert isinstance(inputs, str)
        inputs = [inputs]
    else:
        assert isinstance(inputs, list)

    if tokenizer:
        for x in inputs:
            assert isinstance(x, str)

        inputs = [
            pad(config.max_len, clean_str(x, tokenizer=tokenizer), config.pad_token)
            for x in inputs
        ]
    else:
        for x in inputs:
            assert isinstance(x, list)

        inputs = [pad(config.max_len, x, config.pad_token) for x in inputs]
        # print(inputs)

    if config.use_BERT:
        inputs, masks = [
            list(x) for x in zip(*[bert_wrapper.pre_pro(t) for t in inputs])
        ]
        inputs, masks = torch.tensor(inputs), torch.tensor(masks)
        masks = masks.cuda() if config.gpu else masks
    else:
        inputs = torch.tensor(
            [prep_seq(x, word_to_idx, config.unk_token) for x in inputs],
            dtype=torch.int64,
        )
        masks = None

    inputs = inputs.cuda() if config.gpu else inputs

    with torch.no_grad():
        if config.use_BERT:
            outputs = model(inputs, token_type_ids=None, attention_mask=masks)
            outputs = outputs.logits
        else:
            outputs = model(inputs)

    outputs = softmax(outputs)
    probs = outputs.cpu().detach().numpy().tolist()
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().detach().numpy().tolist()

    if single:
        preds, probs = preds[0], probs[0]

    if val:
        return preds, outputs
    else:
        return preds, probs

class CustomPytorchCNNWrapper(ModelWrapper):
    def __init__(self, model, config, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        # BERT-Attack
        # self.device = torch.device('cuda:1')
        # TextFooler and TextBugger
        # self.device = torch.device('cuda')
        # self.max_length = 40 if args.sst else 256
        self.config = config 
        self.tokenizer = tokenizer


    def __call__(self, text_input_list):

        # text_input_list = [text.split(' ') for text in text_input_list]
        # prediction = self.model.text_pred(text_input_list)
        # all_input_ids = self.encode_fn(self.tokenizer, text_input_list)
        # dataset = TensorDataset(all_input_ids)
        # pred_dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        # self.model.to(self.device)
        # self.model.eval()
        # prediction = []
        # for batch in pred_dataloader:
        #     outputs = self.model(batch[0].to(self.device), token_type_ids=None, attention_mask=(batch[0]>0).to(self.device))
        #     logits = outputs[0]
        #     logits = logits.detach().cpu().numpy()
        #     prediction.append(logits)
        text_input_list = [[w.text.lower() for w in self.tokenizer(t.strip())] for t in text_input_list]
        _, probs = inference(text_input_list, self.model, self.model.w2i, self.config)
        # return np.concatenate(prediction, axis=0)
        return probs

    # def encode_fn(self, tokenizer, text_list):
    #     all_input_ids = []    
    #     for text in text_list:
    #         input_ids = self.tokenizer.encode(
    #                         text,
    #                         truncation=True,                       
    #                         add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
    #                         max_length = self.max_length,           # 设定最大文本长度 200 for IMDb and 40 for sst
    #                         # pad_to_max_length = True,   # pad到最大的长度  
    #                         padding = 'max_length',
    #                         return_tensors = 'pt'       # 返回的类型为pytorch tensor
    #                    )
    #         all_input_ids.append(input_ids)    
    #     all_input_ids = torch.cat(all_input_ids, dim=0)
    #     return all_input_ids

if __name__ == '__main__':
    config = Config()
    config.noise = False
    model = load_for_textattack(config)
    # inputs = [['i', 'love', 'this', 'movie'], ['i', 'hate', 'this', 'movie', 'a']]
    # preds, probs = inference(inputs, model, model.w2i, config)
    # print(preds, probs)

    nlp = English()
    spacy_tokenizer = nlp.tokenizer

    model_wrapper = CustomPytorchCNNWrapper(model, config, spacy_tokenizer)
    dataset = HuggingFaceDataset("imdb", None, "test")
    attack = PWWSRen2019.build(model_wrapper)
    attack_args = AttackArgs(num_examples=1000, checkpoint_dir="checkpoints", shuffle=True)
    attacker = Attacker(attack, dataset, attack_args)

    attacker.attack_dataset()
# train without noise
    # PWWS-100 w/o noise
# +-------------------------------+---------+
# | Attack Results                |         |
# +-------------------------------+---------+
# | Number of successful attacks: | 79      |
# | Number of failed attacks:     | 1       |
# | Number of skipped attacks:    | 20      |
# | Original accuracy:            | 80.0%   |
# | Accuracy under attack:        | 1.0%    |
# | Attack success rate:          | 98.75%  |
# | Average perturbed word %:     | 3.32%   |
# | Average num. words per input: | 216.96  |
# | Avg num queries:              | 1415.45 |
# +-------------------------------+---------+

    # PWWS-100 w noise
# +-------------------------------+--------+
# | Attack Results                |        |
# +-------------------------------+--------+
# | Number of successful attacks: | 38     |
# | Number of failed attacks:     | 31     |
# | Number of skipped attacks:    | 31     |
# | Original accuracy:            | 69.0%  |
# | Accuracy under attack:        | 31.0%  |
# | Attack success rate:          | 55.07% |
# | Average perturbed word %:     | 0.68%  |
# | Average num. words per input: | 216.96 |
# | Avg num queries:              | 1765.8 |
# +-------------------------------+--------+

# train with noise
# PWWS-100 w/o noise
# +-------------------------------+---------+
# | Attack Results                |         |
# +-------------------------------+---------+
# | Number of successful attacks: | 74      |
# | Number of failed attacks:     | 2       |
# | Number of skipped attacks:    | 24      |
# | Original accuracy:            | 76.0%   |
# | Accuracy under attack:        | 2.0%    |
# | Attack success rate:          | 97.37%  |
# | Average perturbed word %:     | 2.44%   |
# | Average num. words per input: | 216.96  |
# | Avg num queries:              | 1400.83 |
# +-------------------------------+---------+

# PWWS-100 w noise
# +-------------------------------+--------+
# | Attack Results                |        |
# +-------------------------------+--------+
# | Number of successful attacks: | 46     |
# | Number of failed attacks:     | 24     |
# | Number of skipped attacks:    | 30     |
# | Original accuracy:            | 70.0%  |
# | Accuracy under attack:        | 24.0%  |
# | Attack success rate:          | 65.71% |
# | Average perturbed word %:     | 0.99%  |
# | Average num. words per input: | 216.96 |
# | Avg num queries:              | 1687.2 |
# +-------------------------------+--------+
