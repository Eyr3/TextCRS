import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import random
from config import Config
from logger import Logger
from data_module import DataModule
from helper import load_data_textatk, bert_freeze_layers, load_ae_data, load_ae_data_chatgpt
from utils import print_model_state_dict
from textattacknew import TrainingArgs, Trainer
from textattacknew.models.wrappers import PyTorchModelWrapper, HuggingFaceModelWrapper
import numpy as np
import torch


def run_train(config, model_wrapper, training_args, logger, data_module):
    print_model_state_dict(logger, model_wrapper.model)

    train_dataset = load_data_textatk(config.model_type, config.dataset, 'train')
    eval_dataset = load_data_textatk(config.model_type, config.dataset, 'val')
    test_dataset = load_data_textatk(config.model_type, config.dataset, 'test')

    trainer = Trainer(model_wrapper, "classification", None, train_dataset, eval_dataset, training_args, data_module)
    trainer.train()

    model_wrapper.model.cuda()
    test = Trainer(model_wrapper, "classification", None, None, test_dataset, training_args, None)
    test.evaluate()


def run_test(config, model_wrapper, training_args):
    if config.ae_data is None:
        test_dataset = load_data_textatk(config.model_type, config.dataset, 'test')
    else:
        test_dataset = load_ae_data(config.ae_data)
    test = Trainer(model_wrapper, "classification", None, None, test_dataset, training_args, None)
    test.evaluate()


def run_certify(config, model_wrapper, training_args, data_module):
    if config.ae_data is None:
        test_dataset = load_data_textatk(config.model_type, config.dataset, 'test')
    else:
        test_dataset = load_ae_data(config.ae_data)
    test = Trainer(model_wrapper, "classification", None, None, test_dataset, training_args, data_module)
    test.certify()


def run_predict(config, model_wrapper, training_args, data_module):
    if config.ae_data is None:
        test_dataset = load_data_textatk(config.model_type, config.dataset, 'test')
    else:
        if 'chatgpt' in config.ae_data:
            test_dataset, id_in_chatgpt = load_ae_data_chatgpt(config.ae_data)
        else:
            test_dataset = load_ae_data(config.ae_data)
        f = open(config.certify_log, 'a+')
        print("====== Predict on {} adversarial examples ======".format(config.atk), file=f, flush=True)
        print("model path: {}".format(config.model_path), file=f, flush=True)
        if 'chatgpt' in config.ae_data: print("id_in_chatgpt={}".format(id_in_chatgpt), file=f, flush=True)
        f.close()

    test = Trainer(model_wrapper, "classification", None, None, test_dataset, training_args, data_module)
    test.predict()


if __name__ == '__main__':
    args = Config()
    logger = Logger(args)

    random.seed(args.seed_val)
    np.random.seed(args.seed_val)
    torch.manual_seed(args.seed_val)
    # spacy.util.fix_random_seed(args.seed_val)

    if args.mode == 'train':
        if args.model_path is None and args.model_type in ['bert', 'newbert']:
            model_path = "bert-base-uncased"
            # model_path = '/data/xinyu/results/fgws/models/bert/{}/noise3_g-n=0.0/best_model'.format(args.dataset)
        else:
            model_path = args.model_path

        output = '{}/{}/'.format(args.save_path, args.name)  # _g={}B, args.glove_size

    elif args.mode == 'test':
        output = '{}/textatk_test/'.format(args.save_path)
        model_path = args.model_path

    elif args.mode in ['certify', 'predict']:
        output = args.model_base_path  # '{}/{}/'.format(args.save_path, args.name)
        model_path = args.model_path

    if args.model_type == 'lstm':
        from textattacknew.models.helpers import LSTMForClassification
        logger.log.info("Loading textattacknew model: LSTMForClassification - {}".format(model_path))
        model = LSTMForClassification(
            dropout=args.dropout_rate,
            max_seq_length=args.max_len,
            num_labels=args.num_classes,
            emb_layer_trainable=False,
            model_path=model_path,
            glove_size=args.glove_size,
        )
        model.cuda()
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    elif args.model_type == 'newlstm':
        from textattacknew.models.helpers import NEWLSTMForClassification
        logger.log.info("Loading textattacknew model: NEWLSTMForClassification - {}".format(model_path))
        model = NEWLSTMForClassification(
            dropout=args.dropout_rate,
            max_seq_length=args.max_len,
            num_labels=args.num_classes,
            emb_layer_trainable=False,
            model_path=model_path,
            glove_size=args.glove_size,
            channel_rate=args.channel_rate,
        )
        model.cuda()
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    elif args.model_type == 'cnn':
        from textattacknew.models.helpers import WordCNNForClassification
        logger.log.info("Loading textattacknew model: WordCNNForClassification - {}".format(model_path))
        model = WordCNNForClassification(
            # dropout=args.dropout_rate,
            max_seq_length=args.max_len,
            num_labels=args.num_classes,
            emb_layer_trainable=False,
            model_path=model_path,
            glove_size=args.glove_size,
        )
        model.cuda()
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    elif args.model_type == 'newcnn':
        from textattacknew.models.helpers import NEWWordCNNForClassification
        logger.log.info("Loading textattacknew model: NEWWordCNNForClassification - {}".format(model_path))
        model = NEWWordCNNForClassification(
            # dropout=args.dropout_rate,
            max_seq_length=args.max_len,
            num_labels=args.num_classes,
            emb_layer_trainable=False,
            model_path=model_path,
            glove_size=args.glove_size,
            channel_rate=args.channel_rate,
        )
        model.cuda()
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    elif args.model_type == 'bert':
        import transformers

        logger.log.info("Loading transformers AutoModelForSequenceClassification")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=args.num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )
        bert_freeze_layers(model, -1)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=args.bert_max_len,
            do_lower_case=True,
        )
        model.cuda()
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    elif args.model_type == 'newbert':
        # https://huggingface.co/transformers/v3.0.2/main_classes/model.html#transformers.PreTrainedModel.from_pretrained            
        logger.log.info("Loading BertForSequenceClassification from new transformers1 - {}".format(model_path))
        from textattacknew.models.transformers1.models.bert import BertForSequenceClassification, BertTokenizer

        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=args.num_classes,
            channel_rate=args.channel_rate,
            output_attentions=False,
            output_hidden_states=False,
            # state_dict=pretrained_dict,
        )
        # https://blog.csdn.net/weixin_42118374/article/details/103761795
        # if args.channel_rate:
        #     channelmodel = BertForSequenceClassification.from_pretrained(
        #         args.channelpath,
        #         num_labels=args.num_classes,
        #         channel_rate=args.channel_rate,
        #         output_attentions=False,
        #         output_hidden_states=False,
        #     )
        #     channelmodel_dict = channelmodel.state_dict()
        #     pretrained_dict = {k: v for k, v in channelmodel_dict.items() if 'channel' in k}
        #     model_dict = model.state_dict()
        #     # # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #     model_dict.update(pretrained_dict)
        #     model.load_state_dict(model_dict)
        #     del channelmodel
        bert_freeze_layers(model, -1)

        tokenizer = BertTokenizer.from_pretrained(
            model_path,
            model_max_length=args.bert_max_len,
            do_lower_case=True,
        )
        model.cuda()
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    if args.mode == 'train':
        ckp_interval_epochs = 5 if 'bert' in args.model_type else 10
        training_args = TrainingArgs(num_epochs=args.num_epoch, start_epoch=args.start_epoch, model_type=args.model_type,
                                     learning_rate=args.learning_rate, checkpoint_interval_epochs=ckp_interval_epochs,
                                     load_best_model_at_end=True, parallel=args.parallel,
                                     per_device_train_batch_size=args.batch_size_train,  # 8,
                                     per_device_eval_batch_size=args.batch_size_val,  # 8,
                                     output_dir=output, log_to_tb=True, tb_log_dir='{}/logs/'.format(output),
                                     if_addnoise=args.if_addnoise,
                                     syn_size=args.syn_size,  # noise1 / 9 / 5 / 6
                                     shuffle_len=args.shuffle_len,  # noise2 / 8
                                     noise_sd=args.noise_sd, dynamic_mu=args.dynamic_mu,  # noise3
                                     use_kl=args.use_kl, alpha=args.CE_loss, VSD_loss=args.VSD_loss,
                                     beta=args.beta,)
    elif args.mode == 'test':
        training_args = TrainingArgs(num_epochs=args.num_epoch, start_epoch=args.start_epoch, model_type=args.model_type,
                                     per_device_eval_batch_size=args.batch_size_val, dynamic_mu=args.dynamic_mu,
                                     output_dir=output, tb_log_dir='{}/logs/'.format(output), test_log=args.test_log)
        # if_addnoise = args.if_addnoise, syn_size = args.syn_size, shuffle_len = args.shuffle_len, noise_sd = args.noise_sd,
    else:
        training_args = TrainingArgs(model_type=args.model_type, num_classes=args.num_classes,
                                     output_dir=output, certify_log=args.certify_log, continue_idx=args.continue_idx,
                                     if_addnoise=args.if_addnoise,
                                     syn_size=int(args.syn_size),  # noise 1 / noise 9
                                     shuffle_len=int(args.shuffle_len),  # noise 2 / noise 8
                                     noise_sd=args.noise_sd,  # noise 3
                                     beta=args.beta, dynamic_mu=args.dynamic_mu,  # noise 4
                                     sigma=args.sigma, skip=args.skip, certify_batch=args.certify_batch,
                                     N=args.N)  # certify

    data_module = DataModule(args, logger) if (args.if_addnoise in [1, 9] and args.mode != 'test') else None
    if args.mode == 'train':
        run_train(args, model_wrapper, training_args, logger, data_module)
    elif args.mode == 'test':
        run_test(args, model_wrapper, training_args)
    elif args.mode == 'predict':
        run_predict(args, model_wrapper, training_args, data_module)
    elif args.mode == 'certify':
        run_certify(args, model_wrapper, training_args, data_module)
