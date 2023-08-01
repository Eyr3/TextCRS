import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random
from config import Config
from logger import Logger
from helper import bert_freeze_layers
from data_module import DataModule
import textattacknew
from textattacknew.models.wrappers import PyTorchModelWrapper, HuggingFaceModelWrapper
from textattacknew.attack_recipes import PWWSRen2019, TextFoolerJin2019, BAEGargInsert2019, InsertWordSyn, InputReductionFeng2018, WordOrderSwap, CLARE2020, GeneticAlgorithmAlzantot2018, BERTAttackLi2020, DeepWordBugGao2018, PSOZang2020, BAEGarg2019, FasterGeneticAlgorithmJia2019, AttacksCombination
from textattacknew.models.helpers import LSTMForClassification, WordCNNForClassification
from utils import load_pkl, shuffle_lists
import time
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle


def run_def(params, logger, data_loader):
    if params.model_type == 'lstm':
        model = LSTMForClassification(
            dropout=params.dropout_rate,
            max_seq_length=params.max_len,
            num_labels=params.num_classes,
            emb_layer_trainable=False,
            model_path=params.model_path,
            glove_size=params.glove_size,
            # if_addnoise=params.if_addnoise,
            # noise_sd=params.noise_sd,
        )
        model.cuda()
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    elif params.model_type == 'cnn':
        model = WordCNNForClassification(
            # dropout=args.dropout_rate,
            max_seq_length=params.max_len,
            num_labels=params.num_classes,
            emb_layer_trainable=False,
            model_path=params.model_path,
            glove_size=params.glove_size,
        )
        model.cuda()
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    elif params.model_type == 'bert':
        import transformers

        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            params.model_path,
            num_labels=params.num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )
        bert_freeze_layers(model, -1)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            params.model_path,
            model_max_length=params.bert_max_len,
            do_lower_case=True,
        )
        model.cuda()
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    logger.log.info("Loading textattacknew model: {}".format(params.model_path))

    new_dataset = []
    test_raw = data_loader.test_raw
    test_pols = data_loader.test_pols
    test_raw, test_pols = shuffle_lists(test_raw, test_pols)
    for i in range(len(test_pols)):
        new_dataset.append((test_raw[i], test_pols[i]))
    dataset = textattacknew.datasets.Dataset(new_dataset)

    style = 'file'  # plain: for replace,
    if params.atk == 'textfooler':
        attack = TextFoolerJin2019.build(model_wrapper)  # Swap words with their 50 closest embedding nearest-neighbors
    elif params.atk == 'swap':
        attack = WordOrderSwap.build(model_wrapper)
    elif params.atk == 'insert':
        attack = InsertWordSyn.build(model_wrapper)
    elif params.atk == 'delete':
        attack = InputReductionFeng2018.build(model_wrapper)
    elif 'bae_i' in params.atk:
        attack = BAEGargInsert2019.build(model_wrapper)
    elif 'clare' in params.atk:  # combination: replace+insert
        attack = CLARE2020.build(model_wrapper)
    elif params.atk == 'combine':
        attack = AttacksCombination.build(model_wrapper)
    elif params.atk == 'cr_ibp':  # compare
        attack = FasterGeneticAlgorithmJia2019.build(model_wrapper)
    elif params.atk == 'psozang':
        attack = PSOZang2020.build(model_wrapper)
    elif params.atk == 'pwws':
        attack = PWWSRen2019.build(model_wrapper)
    elif params.atk == 'bae_r':
        attack = BAEGarg2019.build(model_wrapper)

    # char- textbugger / DeepWordBugGao2018

    attack_args = textattacknew.AttackArgs(
        num_successful_examples=params.num_examples, csv_coloring_style=style, shuffle=True, random_seed=8,
        checkpoint_dir="{}/attacks/checkpoint".format(params.model_root_path),
        log_to_csv='{}/attacks/{}/{}/textatk_{}_{}.csv'.format(params.model_root_path, params.model_type, params.dataset, params.atk, style),  # , params.glove_size
        log_summary_to_json='{}/attacks/{}/{}/textatk_{}_{}.json'.format(params.model_root_path, params.model_type, params.dataset, params.atk, style),  # _{}B , params.glove_size
    )

    attacker = textattacknew.Attacker(attack, dataset, attack_args)
    start = time.time()
    attacker.attack_dataset()
    end = time.time()
    logger.log.info('running time:{}'.format(end-start))


if __name__ == '__main__':
    config = Config()
    random.seed(config.seed_val)
    np.random.seed(config.seed_val)
    torch.manual_seed(config.seed_val)
    logger = Logger(config)
    data_module = DataModule(config, logger)
    config.vocab_size = len(data_module.vocab)
    run_def(config, logger, data_module)


# --parallel  !!
# textatk_attack -model_type lstm -num_examples 1000 -glove_size 42 -mode test
# textattacknew attack --model lstm-imdb --recipe textfooler --num-examples 10
# textattacknew train --model-name-or-path lstm --dataset imdb  --epochs 50 --learning-rate 1e-5
# textattacknew train --model-name-or-path bert-base-cased --dataset glue^stsb  --epochs 3 --learning-rate 1e-5

# CUDA_VISIBLE_DEVICES=5 python textatk_attack.py  --atk 1 --path ./model/sst/0.pt --model_type bert --sst

# CUDA_VISIBLE_DEVICES=1 python textatk_attack.py  --atk 1 --model_type bert --path /data/ZQData/github-oa/fgws-main/data/models/roberta/imdb/checkpoints/e_best/model.pth
    
# CUDA_VISIBLE_DEVICES=1 python textatk_attack.py --path /data/ZQData/github-oa/gaus/model/imdb/roberta/orig/best.pt >> /data/ZQData/github-oa/gaus/atk_re/sst/textfooler/roberta/orig_model.log 2>&1

# CUDA_VISIBLE_DEVICES=5 python textatk_attack.py --path /data/xinyu/results/fgws/data/models/lstm/imdb/ckpt_lr=0.001_de=0 >> /data/ZQData/github-oa/gaus/atk_re/sst/textfooler/roberta/gaus_model.log 2>&1

# CUDA_VISIBLE_DEVICES=4 python textatk_attack.py --path /data/ZQData/github-oa/gaus/model/imdb/roberta/orig/best.pt >> /data/ZQData/github-oa/gaus/atk_re/sst/textfooler/roberta/g_orig_model.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python textatk_attack.py --path /data/ZQData/github-oa/gaus/model/imdb/roberta/gaus/best.pt >> /data/ZQData/github-oa/gaus/atk_re/sst/textfooler/roberta/g_gaus_model.log 2>&1


#---------------imdb roberta--------------------------

# CUDA_VISIBLE_DEVICES=1 python textatk_attack.py --path /data/ZQData/github-oa/fgws-main/data/models/roberta/imdb/checkpoints/e_best/model.pth >> /data/ZQData/github-oa/gaus/atk_re/imdb/textbugger/roberta/orig_model.log 2>&1

# CUDA_VISIBLE_DEVICES=4 python textatk_attack.py --path /data/ZQData/github-oa/fgws-main/data/models/roberta-G/imdb/checkpoints/e_best/model.pth >> /data/ZQData/github-oa/gaus/atk_re/imdb/textbugger/roberta/gaus_model.log 2>&1

# CUDA_VISIBLE_DEVICES=5 python textatk_attack.py --path /data/ZQData/github-oa/fgws-main/data/models/roberta/imdb/checkpoints/e_best/model.pth >> /data/ZQData/github-oa/gaus/atk_re/imdb/textbugger/roberta/g_orig_model.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python textatk_attack.py --path /data/ZQData/github-oa/fgws-main/data/models/roberta-G/imdb/checkpoints/e_best/model.pth >> /data/ZQData/github-oa/gaus/atk_re/imdb/textbugger/roberta/g_gaus_model.log 2>&1


# CUDA_VISIBLE_DEVICES=3 python textatk_attack.py --path /data/ZQData/github-oa/fgws-main/data/models/roberta/amazon/checkpoints/e_best/model.pth >> /data/ZQData/github-oa/gaus/atk_re/amazon/textfooler/roberta-G/step_orig/test_1.log 2>&1
