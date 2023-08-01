
import argparse
import os
from noises.staircase import Staircase
from helper import set_param, set_batch_size, set_lr
import getpass
import numpy as np
import pandas as pd


class Config:
    parser = argparse.ArgumentParser(description="args for experiments")
    parser.add_argument(
        "-mode",
        default="train",
        type=str,
        help="mode is either train, test, attack or detect",
    )
    parser.add_argument(
        "-model_type",
        default="lstm",
        type=str,
        help="one of cnn, lstm, roberta, cnn-G, lstm-G, roberta-G, bert, newbert, bert-G",
    )
    parser.add_argument(
        "-attack",
        default="random",
        type=str,
        help="one of random, prioritized, genetic, pwws",
    )
    parser.add_argument(
        "-visible_devices",
        default="6,7",
        type=str,
        help="which GPUs to use",
    )
    parser.add_argument(
        "-limit",
        default="0",
        type=int,
        help="truncates the respective dataset to a limited size",
    )
    parser.add_argument(
        "-num_epoch",
        default="10",
        type=int,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "-dataset",
        default="imdb",
        type=str,
        help="dataset to use; one of imdb, sst2, amazon, agnews",
    )
    parser.add_argument(
        "-fp_threshold",
        default="0.9",
        type=float,
        help="false positive threshold",
    )
    parser.add_argument(
        "-delta_thr",
        default="None",
        type=str,
        help="delta threshold for detection",
    )
    parser.add_argument(
        "-gpu",
        default=True,
        type=bool,
        help="flag indicating whether to use GPU",
    )
    parser.add_argument(
        "-parallel",
        action="store_true",
        # default=False,
        help="If set, run training on multiple GPUs.",
    )
    parser.add_argument(
        "-attack_train_set",
        action="store_true",
        help="whether to attack the train set",
    )
    parser.add_argument(
        "-attack_val_set",
        action="store_true",
        help="whether to attack the val set",
    )
    parser.add_argument(
        "-detect_val_set",
        action="store_true",
        help="whether to detect adversarial examples on the validation set",
    )
    parser.add_argument(
        "-test_on_val",
        action="store_true",
        help="whether to test on validation set",
    )
    parser.add_argument(
        "-detect_baseline",
        action="store_true",
        help="whether to use baseline detection",
    )
    parser.add_argument(
        "-tune_delta_on_val",
        action="store_true",
        help="use to tune delta on the validation set",
    )
    parser.add_argument(
        "-model_path",
        default=None,
        type=str,
        help="test model path",
    )
    parser.add_argument(
        "-name",
        default="",
        type=str,
        help="save model name",
    )
    parser.add_argument(
        "-lr",
        default="1e-3",
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "-freeze_layer",
        default="-1",
        type=int,
        help="freeze layer count. -1: emb, 0: unfreeze, 1-11: other layers",
    )
    parser.add_argument(
        "-if_addnoise",
        default="0",
        type=int,
        help="0_no noise; 1_synonyms replace; 2_shuffling noise; 3_add word; 4_delete word; 9_SAFER.",
    )
    parser.add_argument(
        "-syn_size",
        default="250",
        type=int,
        help="the size of synonym",
    )
    parser.add_argument(
        "-shuffle_len",
        default="1",
        type=int,
        help="the length of the shuffling range",
    )
    parser.add_argument(
        "-noise_sd",
        default="0",
        type=float,
        help="gaussian noise",
    )
    parser.add_argument(
        "-beta",
        default="0",
        type=float,
        help="each word has the probability of beta to remain unchanged",
    )
    parser.add_argument("-dynamic_mu", type=bool, default=0, help="if use dynamic mu")
    parser.add_argument("-use_kl", type=bool, default=0, help="if use kl divergence to optimize CompressChannel")
    parser.add_argument("-CE_loss", type=float, default=1, help="the weight of CE Loss")
    parser.add_argument("-VSD_loss", type=float, default=0, help="the weight of VSD Loss")
    parser.add_argument("-atk", type=str, default='textfooler', help="attack types")
    parser.add_argument("-num_examples", type=int, default=1000)
    parser.add_argument("-glove_size", type=int, default="42", help="glove size: 6(B) or 42(B)")
    parser.add_argument("-channel_rate", type=float, default=0, help="channel compress layer rate")
    parser.add_argument("-start_epoch", type=int, default=1)
    parser.add_argument("-ae_data", type=str, default=None, help="adversarial example data path")
    # certify
    parser.add_argument("-sigma", type=float, default=0.1, help="certify sigma")
    parser.add_argument("-N0", type=int, default=100)
    parser.add_argument("-N", type=int, default=100000, help="number of samples to certify")
    parser.add_argument("-continue_idx", type=int, default=-1, help="idx continue to certify")
    # parser.add_argument("-alpha", type=float, default=0.001, help="failure probability")
    args = parser.parse_args()

    tune_delta_on_val = args.tune_delta_on_val
    detect_val_set = True if tune_delta_on_val else args.detect_val_set
    attack_val_set = args.attack_val_set
    test_on_val = args.test_on_val
    detect_baseline = args.detect_baseline
    fp_threshold = args.fp_threshold
    delta_thr = None if args.delta_thr == "None" else int(args.delta_thr)
    model_type = args.model_type
    attack = args.attack
    limit = args.limit
    gpu = args.gpu
    parallel = args.parallel
    visible_devices = args.visible_devices
    mode = args.mode
    dataset = args.dataset
    attack_train_set = args.attack_train_set
    # learning_rate = args.lr
    # num_epoch = args.num_epoch
    if_addnoise = args.if_addnoise
    shuffle_len = args.shuffle_len
    decay_epoch = 0  # setting the decay epoch
    weight_decay = 0.01  # weight decay for AdamW
    name = args.name
    model_path = args.model_path
    freeze_layer = args.freeze_layer
    glove_size = args.glove_size
    syn_size = args.syn_size
    noise_sd = args.noise_sd
    use_kl = args.use_kl
    CE_loss = args.CE_loss
    VSD_loss = args.VSD_loss
    beta = args.beta
    atk = args.atk
    num_examples = args.num_examples
    start_epoch = args.start_epoch
    ae_data = args.ae_data
    sigma = args.sigma
    N = args.N
    num_epoch, learning_rate = set_lr(model_type)
    mu_sigma = None
    mu = 0
    channel_rate = args.channel_rate
    dynamic_mu = args.dynamic_mu
    continue_idx = args.continue_idx

    staircase_mech = None if if_addnoise != 1 else Staircase(epsilon=5 / syn_size, gamma=1, sensitivity=1, random_state=1)
    use_BERT = "bert" in model_type or model_type == "bert-G" or model_type == "roberta" or model_type == "roberta-G"
    if model_type in ['newlstm', 'newbert', 'newcnn']:
        if use_kl:
            name = '{}VSD{}_'.format(name, VSD_loss)
        if dynamic_mu:
            name = '{}mu'.format(name)
        name = '{}channel_{}'.format(name, channel_rate)

    # Training params
    max_len, bert_max_len, num_classes, val_split_size = set_param(dataset)
    if if_addnoise in [4, 7]:
        shuffle_len = bert_max_len if use_BERT else max_len

    if if_addnoise in [1, 9, 5, 6]:  # 9_SAFER
        name = '{}noise{}_k={}'.format(name, if_addnoise, syn_size)
        noise = syn_size
    elif if_addnoise in [2, 8]:  # 8_shuffle after padding
        name = '{}noise{}_sh-len={}'.format(name, if_addnoise, shuffle_len)
        noise = shuffle_len
    elif if_addnoise in [3, 7]:
        name = '{}noise{}_g-n={}'.format(name, if_addnoise, noise_sd)
        noise = noise_sd
        # if 'lstm' in model_type:
        #     mu_sigma = np.load('/data/xinyu/results/fgws/data/pretrained/gloVe/glove_data/mu_sigma_42B.npy')
        #     mu = list(mu_sigma[:, 0])
        #     print('mean mu: {}, minimum sigma: {}'.format(np.mean(mu_sigma[:, 0]), np.min(mu_sigma[:, 1])))
    elif if_addnoise == 4:  # Bernoulli distribution + shuffle_len
        name = '{}noise{}_beta={}_sh-len={}'.format(name, if_addnoise, beta, shuffle_len)
        noise = beta
    elif if_addnoise != 0:
        name = '{}noise{}'.format(name, if_addnoise)

    # max_len = 50 if dataset == "sst2" else 200
    # bert_max_len = 256 if dataset == "imdb" else 128
    # bert_max_len = 40 if dataset == "sst2" else 256

    embed_size = 300
    batch_size_train, batch_size_val, batch_size_test = set_batch_size(use_BERT, dataset)

    if mode == 'certify':
        certify_batch = 2000  #if dataset == 'agnews' else 500
    else:
        certify_batch = 500

    if dataset == 'agnews': skip = 15  # 25, for certify about 310 samples
    elif dataset == 'amazon': skip = 100  # 160
    elif dataset == 'imdb': skip = 50  # 50, 80

    # Params for CNN
    filter_sizes = [3, 4, 5]
    stride = 1
    num_feature_maps = 1005

    # Params for LSTM
    hidden_size = 150
    num_layers = 1
    rnn_size = 10
    lr_decay = 0.9
    dropout_rate = 0  # .3 if dataset == 'amazon' else 0

    # Params for RoBERTa
    if use_BERT:
        # RoBERTa params from https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md
        warmup_percent = 0.06
        adam_eps = 1e-8
        clip_norm = 0.0

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    project_root_path = "."
    # base_path = os.path.dirname(__file__)
    data_root_path = "/data/xinyu/results/fgws/"  # /data/xinyu/results/fgws/, /home/zhangxinyu/data/fgws/
    model_root_path = "/data/xinyu/results/fgws/"
    save_path = '{}/models/{}/{}/'.format(model_root_path, model_type, dataset)
    model_name_suffix = ""
    if ae_data is not None:  # predict using adversarial text
        ae_data_model = 'lstm' if 'lstm' in model_type else 'cnn' if 'cnn' in model_type else 'bert'
        ae_data = '{}/attacks/{}/{}/textatk_{}.csv'.format(model_root_path, ae_data_model, dataset, ae_data)
        
        skip = 1 if 'chatgpt' in ae_data or 'clare' in ae_data or ('bae_i' in ae_data and dataset == 'imdb') else 2

    if mode == "train":
        model_name_suffix = "{}/{}".format(model_type, dataset)
        # name = "{}_lr={}".format(name, learning_rate)
        if channel_rate:
            channelpath = '{}/models/newbert/{}/dynamicchannel_{}noise3_g-n=1.0/best_model/'. \
                format(model_root_path, dataset, channel_rate)

    elif mode == "test":
        model_name_suffix = "test/{}/{}".format(model_type, dataset)
        if model_path is None:  # 若不指定model,则选择clean model中acc最高的
            if len(name) == 0: name = 'textatk'
            if args.model_type in ['lstm', 'newlstm', 'cnn', 'newcnn']:
                model_path = '{}/best/best_model/'.format(save_path, name, learning_rate)  # {}_lr={}_g=42B
            elif args.model_type in ['bert', 'newbert']:
                model_path = '{}/{}_lr={}_freeze/best_model/'.format(save_path, name, learning_rate)

    elif mode in ['certify', 'predict']:
        if getpass.getuser() == "huangpeng":
            model_name_suffix = "/{}/{}/{}/noise{}/".format(mode, model_type, dataset, if_addnoise)
        else:
            model_name_suffix = "/{}1/{}/{}/noise{}/".format(mode, model_type, dataset, if_addnoise)

        if model_path is None:
            model_path = "{}/{}/best_model".format(save_path, name)

    elif mode == "attack":
        attack_set = (
            "train_set"
            if attack_train_set
            else "val_set"
            if attack_val_set
            else "test_set"
        )
        model_name_suffix = "limit_{}/{}/{}/{}/{}".format(
            limit, model_type, attack, dataset, attack_set
        )
    elif mode == "detect":
        attack_set = (
            "train_set"
            if attack_train_set
            else "val_set"
            if attack_val_set or tune_delta_on_val
            else "test_set"
        )

        model_name_suffix = "{}_attack_{}_{}_{}{}".format(
            attack,
            model_type,
            dataset,
            attack_set,
            "_limit_{}".format(int(limit)) if int(limit) > 0 else "",
        )

        model_name_suffix += "_fp_threshold_{}".format(fp_threshold)

        if delta_thr is not None:
            model_name_suffix += "_delta_thr_{}".format(delta_thr)

        if detect_baseline:
            model_name_suffix += "_detect_baseline"

    model_dir = (
        # "smooth" if mode == "test" and ae_data is not None
        "models" if mode in ["train", "test"]
        else "attacks" if mode == "attack"
        else "smooth" if mode in ["certify", "predict"]
        else "detections"
    )

    model_base_path = "{}/{}/{}".format(model_root_path, model_dir, model_name_suffix)

    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path)

    if mode == "train":
        tb_log_train_path = "{}/{}/logs/tb_train".format(model_base_path, name)
        tb_log_val_path = "{}/{}/logs/tb_val".format(model_base_path, name)

        if not os.path.exists(tb_log_train_path):
            os.makedirs(tb_log_train_path)

        if not os.path.exists(tb_log_val_path):
            os.makedirs(tb_log_val_path)

    if mode == 'certify':
        # if 'best_model' in model_base_path:
        certify_log = "{}/from0.1_noise_{}_sigma_{}".format(model_base_path, noise, sigma)  # _N=200000_bestmodel_from370
        # else:
        # certify_log = "{}/last_model_noise_{}_sigma_{}".format(model_base_path, noise, sigma)
        if VSD_loss:
            certify_log = '{}_VSD_{}'.format(certify_log, VSD_loss)
        if channel_rate:
            certify_log = '{}_channel_{}'.format(certify_log, channel_rate)
        if dynamic_mu:
            certify_log = '{}_mu'.format(certify_log)
        if ae_data is not None:  # certify ae_data
            certify_log = '{}_{}'.format(certify_log, atk)
        if not os.path.exists(certify_log):
            os.system(r"touch {}".format(certify_log))
        else:
            try:
                data = pd.read_table(os.path.join(certify_log), sep='\t')
                if len(data) <= 1:
                    continue_idx = -1
                else:
                    continue_idx = data['idx'].values[-1]
            except:
                continue_idx = -1
    if mode == 'predict':
        if ae_data is None:
            certify_log = "{}/benign_noise_{}_sigma_{}".format(model_base_path, noise, sigma)  # certified benign accuracy
        else:
            certify_log = "{}/noise_{}_sigma_{}_{}".format(model_base_path, noise, sigma, atk)
        if VSD_loss:
            certify_log = '{}_VSD_{}'.format(certify_log, VSD_loss)
        if channel_rate:
            certify_log = '{}_channel_{}'.format(certify_log, channel_rate)
        if dynamic_mu:
            certify_log = '{}_mu'.format(certify_log)
        if not os.path.exists(certify_log):
            os.system(r"touch {}".format(certify_log))
        f = open(certify_log, 'a+')
        print('ae_data={}'.format(ae_data), file=f, flush=True)
        f.close()

    if mode == 'test':
        test_log = None
        if ae_data is not None:
            test_log = "{}/noise_{}_sigma_{}_{}".format(model_base_path, noise, sigma, atk)
            if not os.path.exists(test_log):
                os.system(r"touch {}".format(test_log))
            f = open(test_log, 'a+')
            print('ae_data={}'.format(ae_data), file=f, flush=True)
            f.close()

    cf_path = "{}/data/pretrained/counter-fitted/counter-fitted-vectors.txt".format(data_root_path)
    glove_path = "/data/xinyu/results/fgws/data/pretrained/gloVe/glove.42B.300d.txt".format(data_root_path)  # 42B.300d

    # LSTM
    path_to_pre_trained_init = "{}/data/{}/{}_pretrained_init.npy".format(data_root_path, dataset, max_len)
    # bert
    if use_BERT:
        path_to_pre_trained_init = "{}/data/{}/{}_pretrained_init.npy".format(data_root_path, dataset, bert_max_len)
    # path_to_pre_trained_init = "{}/data/{}/pretrained_init.npy".format(data_root_path, dataset)

    keep_embeddings_fixed = True

    path_to_imdb = "{}/data/imdb".format(data_root_path)
    path_to_sst2 = "{}/data/sst2/tsv-format".format(data_root_path)
    path_to_amazon = "{}/data/amazon".format(data_root_path)  #_reviews
    path_to_agnews = "{}/data/agnews".format(data_root_path)

    seed_val = 768
    vocab_size = 0

    path_to_dist_mat = data_root_path + "/data/{}/dist_mat.npy".format(dataset)  #
    path_to_dist_mat_neighbor = data_root_path + "/data/{}/dist_mat_neighbor.npy".format(dataset)  #
    path_to_idx_to_dist_mat_idx_dict = (data_root_path + "/data/{}/idx_to_dist_mat_idx.pkl".format(dataset))  #
    path_to_dist_mat_idx_to_idx_dict = (data_root_path + "/data/{}/dist_mat_idx_to_idx.pkl".format(dataset))  #

    restore_model_path = "{}/models/{}/{}".format(model_root_path, model_type, dataset)
    load_model_path = "/{}/{}/e_best/model.pth".format(restore_model_path, name)

# Params for Genetic attack
    k = 8
    LM_cut = 4
    LM_window_size = 5
    max_alt_ratio = 0.2
    dist_metric = "euclidean"
    # delta = 0.5 if dataset == "imdb" else 1.0
    delta = 1.0 if dataset == "sst" else 0.5
    num_pop = 60
    num_gen = 20
    path_to_attack_dist_mat = "{}/data/{}/attack_dist_mat_{}.npy".format(
        model_root_path, dataset, dist_metric
    )
    path_to_attack_dist_mat_word_to_idx = (
        "{}/data/{}/attack_dist_mat_word_to_idx.pkl".format(model_root_path, dataset)
    )
    path_to_attack_dist_mat_idx_to_word = (
        "{}/data/{}/attack_dist_mat_idx_to_word.pkl".format(model_root_path, dataset)
    )

    bootstrap_sample_size = 10000
    ci_alpha = 0.01

    pad_token = "<pad>"
    eos_token = "."
    unk_token = "<unk>"

    restore_delta_path = "{}/detections/delta_experiments/deltas/{}".format(
        model_root_path, "cont{}".format(fp_threshold)
    )

    if tune_delta_on_val and not os.path.exists(restore_delta_path):
        os.makedirs(restore_delta_path)

    restore_delta_path += "/{}_{}.pkl".format(model_type, dataset)
