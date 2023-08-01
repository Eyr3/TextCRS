"""
Parts based on https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX
"""
import re
import torch
import pickle
import math
import random
import sklearn
import os
import torch.nn as nn
import numpy as np
import statsmodels.stats.api as stats
from shutil import copyfile
from nltk.corpus import wordnet
from sklearn.metrics import roc_auc_score


class AttackArgs:
    def __init__(self, config, logger, model_word_to_idx, data_tokenizer):
        self.config = config
        self.logger = logger
        self.model_word_to_idx = model_word_to_idx
        self.data_tokenizer = data_tokenizer


def compute_accuracy(preds, labels):
    assert len(preds) == len(labels)
    return len([True for p, t in zip(preds, labels) if p == t]) / len(preds)


def load_model(path, model, logger):
    logger.log.info("Load model from {}".format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def get_word_net_synonyms(word):
    """
    Parts from https://github.com/JHL-HUST/PWWS/blob/master/paraphrase.py
    """
    synonyms = []

    for synset in wordnet.synsets(word):
        for w in synset.lemmas():
            synonyms.append(w.name().replace("_", " "))

    synonyms = sorted(
        list(set([x.lower() for x in synonyms if len(x.split()) == 1]) - {word})
    )

    return synonyms


def compute_perplexity_GPT2(inputs, tokenizer, language_model, is_gpu=True):
    """
    Following https://github.com/huggingface/pytorch-transformers/issues/473
    """
    tokens = torch.tensor(
        [tokenizer.encode(i, add_special_tokens=True) for i in inputs]
    )

    if is_gpu:
        tokens = tokens.cuda()

    with torch.no_grad():
        output = language_model(tokens, labels=tokens)

    return math.exp(output.loss.item())


def get_freq(freq, word, use_log=False):
    try:
        return np.log(1 + freq[word]) if use_log else freq[word]
    except KeyError:
        return 0


def get_n_neighbors_delta(word, attack_word_to_idx, dist_mat, delta, n_neighbors_delta_map):
    try:
        return n_neighbors_delta_map[word]
    except KeyError:
        w_idx = attack_word_to_idx[word]
        v_neighbors = [x for x in dist_mat[w_idx] if x <= delta]
        n_neighbors_delta_map[word] = len(v_neighbors)
        return n_neighbors_delta_map[word]


def attack_get_neighboring_embeddings(word, dist_mat, attack_word_to_idx, attack_idx_to_word, k, orig_input, delta=None):
    w_idx = attack_word_to_idx[word]
    orig_idx = attack_word_to_idx[orig_input]
    neighbors = dist_mat[w_idx]
    sorted_idx = np.argsort(neighbors)
    sorted_idx = sorted_idx[: k + 2].tolist()
    sorted_idx = [i for i in sorted_idx if i not in [w_idx, orig_idx]][:k]
    sorted_idx = (
        [i for i in sorted_idx if neighbors[i] <= delta]
        if delta is not None
        else sorted_idx
    )
    w_neighbors = [attack_idx_to_word[i] for i in sorted_idx]

    return w_neighbors


def get_neighboring_embeddings(word, dist_mat, word_to_idx, idx_to_word, k, dist_mat_idx_to_idx, idx_to_dist_mat_idx):
    idx = idx_to_dist_mat_idx[word_to_idx[word]]
    neighbors = dist_mat[idx]
    sorted_idx = np.argsort(neighbors).tolist()
    # sorted_idx.remove(idx)
    verb_neighbors = [idx_to_word[dist_mat_idx_to_idx[i]] for i in sorted_idx[:k]]

    return verb_neighbors


def clean_str(string, tokenizer=None):
    """
    Parts adapted from https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/mydatasets.py
    """
    assert isinstance(string, str)

    string = string.replace("<br />", "")
    string = re.sub(r"[^a-zA-Z0-9.]+", " ", string)

    return (
        string.strip().lower().split()
        if tokenizer is None
        else [t.text.lower() for t in tokenizer(string.strip())]
    )


def save_pkl(file, path):
    with open(path, "wb") as handle:
        pickle.dump(file, handle)


def load_pkl(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def prep_seq(seq, word_to_idx, unk_token):
    assert isinstance(seq, list)
    seq_num = []

    for word in seq:
        try:
            seq_num.append(word_to_idx[word])
        except KeyError:
            seq_num.append(word_to_idx[unk_token])

    return seq_num


def pad(max_len, seq, token):
    assert isinstance(seq, list)
    abs_len = len(seq)

    if abs_len > max_len:
        seq = seq[:max_len]
    else:
        seq += [token] * (max_len - abs_len)

    return seq


def cut_raw(seq, max_len):
    assert isinstance(seq, list)
    return seq[:max_len]


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
            '''
            #测试&攻击加噪inputs
            embeds_init = getattr(model, 'roberta').embeddings.word_embeddings(inputs)
            #print(f'the shape of embeds:{embeds_init.shape} '){{16,256,768}}
            #print(f'the example of embeds:{embeds_init[0]} ')
            
            device=torch.device("cuda")
            step = 0.1
            gaussian = step * np.random.normal(0,1,(embeds_init.shape))
            g = torch.tensor(gaussian).to(device)
            embeddings = embeds_init + g.float()
            
            outputs = model(token_type_ids=None, attention_mask=masks,inputs_embeds=embeddings)
            '''

            #原始inputs
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


def crossover(parent_1, parent_2):
    new_seq = []
    idx = []

    for i in range(len(parent_1.text)):
        if random.random() > 0.5:
            new_seq.append(parent_1.text[i])

            if i in parent_1.perturbed_idx:
                idx.append(i)
        else:
            new_seq.append(parent_2.text[i])

            if i in parent_2.perturbed_idx:
                idx.append(i)

    return new_seq, idx


def shuffle_lists(*args):
    """
    See https://stackoverflow.com/a/36695026
    """
    zipped = list(zip(*args))
    random.shuffle(zipped)
    return [list(x) for x in zip(*zipped)]


def bootstrap_sample(all_unperturbed, all_perturbed, bootstrap_sample_size=2000):
    scores_sum = {}
    perturbed_auc_scores = [score for score, _ in all_perturbed]
    perturbed_auc_labels = [1] * len(perturbed_auc_scores)
    unperturbed_auc_labels = [0] * len(perturbed_auc_scores)
    pos = len(all_perturbed)
    t_p = [l for _, l in all_perturbed].count(1)
    f_n = pos - t_p

    for _ in range(bootstrap_sample_size):
        neg = pos
        sample = random.sample(all_unperturbed, neg)
        f_p = [l for _, l in sample].count(1)
        t_n = neg - f_p
        unperturbed_auc_scores = [score for score, _ in sample]

        scores = compute_scores(
            perturbed_auc_scores + unperturbed_auc_scores,
            perturbed_auc_labels + unperturbed_auc_labels,
            pos,
            neg,
            t_p,
            t_n,
            f_p,
            f_n,
        )

        for name, score in scores.items():
            try:
                scores_sum[name].append(score)
            except KeyError:
                scores_sum[name] = [score]

    return scores_sum


def get_ci(data, alpha=0.05):
    return stats.DescrStatsW(data).tconfint_mean(alpha=alpha)


def compute_scores(probs_one, labels, pos, neg, t_p, t_n, f_p, f_n, round_scores=False):
    assert t_p + f_n == pos
    assert t_n + f_p == neg
    assert len(probs_one) == pos + neg == len(labels)

    scores = {
        "auc": roc_auc_score(labels, probs_one),
        "tpr": t_p / pos if pos > 0 else 0,
        "fpr": f_p / neg if neg > 0 else 0,
        "tnr": t_n / neg if neg > 0 else 0,
        "fnr": f_n / pos if pos > 0 else 0,
        "pr": t_p / (t_p + f_p) if t_p + f_p > 0 else 0,
        "re": t_p / (t_p + f_n) if t_p + f_n > 0 else 0,
        "f1": (2 * t_p) / (2 * t_p + f_p + f_n) if 2 * t_p + f_p + f_n > 0 else 0,
        "acc": (t_p + t_n) / (pos + neg) if pos + neg > 0 else 0,
    }

    if round_scores:
        scores = {k: np.round(v * 100, 1) for k, v in scores.items()}

    return scores


def print_model_state_dict(logger, model):
    """
    See https://pytorch.org/tutorials/beginner/saving_loading_models.html
    See https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """
    logger.log.info("Model state dict:")

    for param_tensor in model.state_dict():
        logger.log.info(
            "{}\t{}".format(param_tensor, model.state_dict()[param_tensor].size())
        )

    logger.log.info(
        "Num params: {}".format(
            sum([x.numel() for x in model.parameters() if x.requires_grad])
        )
    )


def attack_time_stats(logger, exec_times, curr_attack_time, num_remaining):
    time_used = sum(exec_times)
    time_remaining = np.mean(exec_times) * num_remaining

    logger.log.info(
        "Total time elapsed: {} secs OR {} mins OR {} hrs".format(
            np.round(time_used, 2),
            np.round(time_used / 60, 2),
            np.round(time_used / 3600, 2),
        )
    )
    logger.log.info(
        "Time needed for this attack: {}".format(np.round(curr_attack_time, 2))
    )
    logger.log.info(
        "Average time per attack so far: {}".format(np.round(np.mean(exec_times), 2))
    )
    logger.log.info(
        "ETA: {} secs OR {} mins OR {} hrs".format(
            np.round(time_remaining, 2),
            np.round(time_remaining / 60, 2),
            np.round((time_remaining / 3600), 2),
        )
    )


def load_attack_mat(config, logger):
    """
    Source: https://github.com/nesl/nlp_adversarial_examples (modified)
    """
    if not os.path.exists(config.path_to_attack_dist_mat):
        attack_dist_mat_word_to_idx = {}
        attack_dist_mat_idx_to_word = {}

        logger.log.info("Loading counter-fitted model for attack")
        f = open(config.cf_path, "r")
        model = []
        model_dict = []
        idx = 0

        for line in f:
            row = line.strip().split(" ")
            word = row[0]

            embedding = np.array([float(val) for val in row[1:]])
            model.append(embedding)
            model_dict.append(word)
            attack_dist_mat_word_to_idx[word] = idx
            attack_dist_mat_idx_to_word[idx] = word
            idx += 1

        logger.log.info("Done. {} words loaded!".format(len(model)))
        logger.log.info("Compute {} dist mat".format(config.dist_metric))

        if config.dist_metric == "euclidean":
            attack_dist_mat = sklearn.metrics.pairwise.euclidean_distances(model)
        else:
            attack_dist_mat = sklearn.metrics.pairwise.cosine_distances(model)

        logger.log.info("Attack dist mat shape {}".format(np.shape(attack_dist_mat)))
        logger.log.info("Save attack mat")

        np.save(config.path_to_attack_dist_mat, attack_dist_mat)
        save_pkl(
            attack_dist_mat_word_to_idx, config.path_to_attack_dist_mat_word_to_idx
        )
        save_pkl(
            attack_dist_mat_idx_to_word, config.path_to_attack_dist_mat_idx_to_word
        )

        logger.log.info("Saved attack mat")
    else:
        logger.log.info(
            "Load pre-computed attack mat from {}".format(
                config.path_to_attack_dist_mat
            )
        )
        attack_dist_mat_word_to_idx = load_pkl(
            config.path_to_attack_dist_mat_word_to_idx
        )
        attack_dist_mat_idx_to_word = load_pkl(
            config.path_to_attack_dist_mat_idx_to_word
        )
        attack_dist_mat = np.load(config.path_to_attack_dist_mat)
        logger.log.info("Attack dist mat shape: {}".format(np.shape(attack_dist_mat)))

    return attack_dist_mat, attack_dist_mat_word_to_idx, attack_dist_mat_idx_to_word


def compute_adversarial_word_overlap(adv_mods, detect_mods, logger):
    mods_idxs = set([i for (_, _, i) in adv_mods])
    t_mods_idxs = set([i for (_, _, i) in detect_mods])

    re_identified = (
        len(list(mods_idxs & t_mods_idxs)) / len(mods_idxs) if len(mods_idxs) > 0 else 0
    )

    pr_identified = (
        len(list(mods_idxs & t_mods_idxs)) / len(t_mods_idxs)
        if len(t_mods_idxs) > 0
        else 0
    )

    logger.log.info("Precision re-identified perturbed idxs: {}".format(pr_identified))
    logger.log.info("Recall re-identified perturbed idxs: {}".format(re_identified))

    return re_identified, pr_identified


def get_attack_data(config, data_module):
    if config.attack_train_set:
        attack_sequences = data_module.train_texts
        attack_pols = data_module.train_pols
    elif config.attack_val_set or config.tune_delta_on_val or config.detect_val_set:
        attack_sequences = data_module.val_texts
        attack_pols = data_module.val_pols
    else:
        attack_sequences = data_module.test_texts
        attack_pols = data_module.test_pols

    if config.limit > 0:
        attack_sequences = attack_sequences[: config.limit]
        attack_pols = attack_pols[: config.limit]

    return attack_sequences, attack_pols


def list_join(input_list):
    assert isinstance(input_list, list)
    return " ".join(input_list)


def copy_file(config):
    copyfile(
        "{}/config.py".format(config.project_root_path),
        "{}/{}/config.py".format(config.model_base_path,config.name),
    )


def get_oov_count(perturbed_indices, word_to_idx):
    oov = 0

    for (_, subst, _) in perturbed_indices:
        try:
            _ = word_to_idx[subst]
        except KeyError:
            oov += 1

    return oov
