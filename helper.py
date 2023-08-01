import copy
import random
import numpy as np
from utils import load_pkl, shuffle_lists, clean_str
import math
import pandas as pd
import torch
from spacy.lang.en import English
from textattacknew.shared import logger, utils
from torch.utils.data import TensorDataset, DataLoader, random_split
from textattacknew.models.wrappers import ModelWrapper
from textattacknew.datasets import Dataset


def adjust_learning_rate(optimizer, lr_decay):
    if optimizer.param_groups[0]['lr'] > 0.0001:
        print("\n--------DECAYING learning rate.--------")
        optimizer.param_groups[0]['lr'] *= lr_decay
        print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
    # for param_group in optimizer.param_groups:
    #     if param_group['lr'] > 0.00001:
    #         print("\n--------DECAYING learning rate.--------")
    #         param_group['lr'] = param_group['lr'] * lr_decay
    #         print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def bert_freeze_layers(model, layer_count):
    # We freeze here the embeddings of the model
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    # for param in model.bert.channel.parameters():
    #     param.requires_grad = False
    if layer_count != -1:
        # if freeze_layer_count == -1, we only freeze the embedding layer
        # otherwise we freeze the first `freeze_layer_count` encoder layers
        for layer in model.bert.encoder.layer[:layer_count]:
            for param in layer.parameters():
                param.requires_grad = False
# ref: https://raphaelb.org/posts/freezing-bert/# https://blog.csdn.net/HUSTHY/article/details/104006106


def addnoise_1(sentences, dist_mat_neighbor, word_to_idx, idx_to_word, dist_mat_idx_to_idx, idx_to_dist_mat_idx, staircase_mech):
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            word = sentences[i][j]
            try:
                idx = idx_to_dist_mat_idx[word_to_idx[word]]
                if idx is not False:
                    embed_neighbors = [idx_to_word[dist_mat_idx_to_idx[i]] for i in
                                       dist_mat_neighbor[idx]]  # dist_mat_neighbor[idx][:syn_size]
                    # embed_neighbors = get_neighboring_embeddings(word, dist_mat, word_to_idx, idx_to_word, syn_size,
                    #                                              dist_mat_idx_to_idx, idx_to_dist_mat_idx)
                    syn_idx = int(staircase_mech.randomise(0))  # random.randint(0, syn_size-1)
                    while syn_idx >= 250:
                        syn_idx = int(staircase_mech.randomise(0))

                    sentences[i][j] = copy.deepcopy(embed_neighbors[syn_idx])
            except Exception as e:
                pass
            continue


def addnoise_2(sentences, shuffle_len):
    for i in range(len(sentences)):
        for j in range(math.ceil(len(sentences[i]) / shuffle_len)):
            part_list = sentences[i][j * shuffle_len: (j + 1) * shuffle_len]
            random.shuffle(part_list)
            sentences[i][j * shuffle_len: (j + 1) * shuffle_len] = part_list


def addnoise_2_bert(input_ids, shuffle_len):
    for i in range(len(input_ids)):
        for j in range(math.ceil(len(input_ids.data['input_ids'][i]) / shuffle_len)):
            input_ids.data['input_ids'][i][j * shuffle_len: (j + 1) * shuffle_len], \
            input_ids.data['token_type_ids'][i][j * shuffle_len: (j + 1) * shuffle_len], \
            input_ids.data['attention_mask'][i][j * shuffle_len: (j + 1) * shuffle_len] = torch.tensor(
                shuffle_lists(input_ids.data['input_ids'][i][j * shuffle_len: (j + 1) * shuffle_len],
                              input_ids.data['token_type_ids'][i][j * shuffle_len: (j + 1) * shuffle_len],
                              input_ids.data['attention_mask'][i][j * shuffle_len: (j + 1) * shuffle_len]))


def addnoise_3(sentences, pad):
    for i in range(len(sentences)):
        sentences[i].insert(random.randint(0, len(sentences[i])), pad)


def addnoise_4(sentences, beta, pad):
    for i in range(len(sentences)):
        max_len = len(sentences[i])
        s = np.random.binomial(1, beta, max_len)  # generate max_len size [0,1] list, the probability of 1 is beta
        pad_position = np.nonzero(s)[0]
        for j in pad_position:
            sentences[i][j] = pad


# def addnoise_4(token, beta, pad_token):
#     for i in range(len(token)):
#         max_len = len(token[i])
#         # s = [random.sample(0, max_len-1) for i in range(int(beta*max_len))]  # randint
#         # for j in s:
#         #     token[i][j] = pad_token
#         s = np.random.binomial(1, beta, max_len)  # generate max_len size [0,1] list, the probability of 1 is beta
#         pad_position = np.nonzero(s)[0]
#         for j in pad_position:
#             token[i][j] = pad_token


def addnoise_9(sentences, dist_mat_neighbor, word_to_idx, idx_to_word, dist_mat_idx_to_idx, idx_to_dist_mat_idx, syn_size):
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            word = sentences[i][j]
            try:
                idx = idx_to_dist_mat_idx[word_to_idx[word]]
                if idx is not False:
                    embed_neighbors = [idx_to_word[dist_mat_idx_to_idx[i]] for i in dist_mat_neighbor[idx]]
                    syn_idx = random.randint(0, syn_size-1)

                    sentences[i][j] = copy.deepcopy(embed_neighbors[syn_idx])
            except Exception as e:
                pass
            continue


def addnoise_5(sentences, nn_matrix, word2index, index2word, staircase_mech):
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            word = sentences[i][j]
            try:
                idx = word2index[word]
                embed_neighbors = [index2word.item()[k] for k in nn_matrix[idx]]
                syn_idx = int(staircase_mech.randomise(0))  # random.randint(0, syn_size-1)
                while syn_idx >= 250:
                    syn_idx = int(staircase_mech.randomise(0))
                sentences[i][j] = embed_neighbors[syn_idx]
            except Exception as e:
                pass
            continue


def addnoise_6(sentences, nn_matrix, word2index, index2word, syn_size):
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            word = sentences[i][j]
            try:
                idx = word2index[word]
                embed_neighbors = [index2word.item()[k] for k in nn_matrix[idx]]
                syn_idx = random.randint(0, syn_size - 1)
                sentences[i][j] = embed_neighbors[syn_idx]
            except Exception as e:
                pass
            continue


def addnoise_3_certify(sentences, pad):
    for i in range(len(sentences)):
        sentences[i][random.randint(0, len(sentences[i])-1)] = pad


def addnoise_4_certify(sentences, beta, pad):
    for i in range(len(sentences)):
        max_len = len(sentences[i])
        s = np.random.binomial(1, beta, max_len)  # generate max_len size [0,1] list, the probability of 1 is beta
        pad_position = np.nonzero(s)[0]
        for j in pad_position[::-1]:
            sentences[i].insert(j, pad)


def addnoise_5_certify(sentences, nn_matrix, word2index, index2word, staircase_mech):
    # Get synonyms matrix
    synonyms_matrix = [[]] * len(sentences[0])
    for i in range(len(sentences[0])):
        word = sentences[0][i]
        try:
            idx = word2index[word]
            synonyms_matrix[i] = [index2word.item()[k] for k in nn_matrix[idx]]
        except Exception as e:
            synonyms_matrix[i] = [word for k in range(250)]

    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            syn_idx = int(staircase_mech.randomise(0))  # random.randint(0, syn_size-1)
            while syn_idx >= 250:
                syn_idx = int(staircase_mech.randomise(0))
            sentences[i][j] = synonyms_matrix[j][syn_idx]


def addnoise_6_certify(sentences, nn_matrix, word2index, index2word, syn_size):
    # Get synonyms matrix
    synonyms_matrix = [[]] * len(sentences[0])
    for i in range(len(sentences[0])):
        word = sentences[0][i]
        try:
            idx = word2index[word]
            synonyms_matrix[i] = [index2word.item()[k] for k in nn_matrix[idx]]
        except Exception as e:
            synonyms_matrix[i] = [word for k in range(250)]

    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            syn_idx = random.randint(0, syn_size - 1)
            sentences[i][j] = synonyms_matrix[j][syn_idx]


def load_data_textatk(model_type, dataset, mode):
    new_dataset = []
    base_path = "/data/xinyu/results/fgws/models/{}/{}/data".format('lstm', dataset)  # model_type
    # raw = list(load_pkl("{}/{}_raw.pkl".format(base_path, mode)))
    texts = list(load_pkl("{}/{}_texts.pkl".format(base_path, mode)))
    raw = [' '.join(texts[i]) for i in range(len(texts))]
    pols = list(load_pkl("{}/{}_pols.pkl".format(base_path, mode)))
    raw, pols = shuffle_lists(raw, pols)
    for i in range(len(pols)):
        new_dataset.append((raw[i], pols[i]))
    logger.info('{} set length: {}'.format(mode, len(new_dataset)))
    new_dataset = Dataset(new_dataset)
    return new_dataset


def load_ae_data(path):
    logger.info('load ae data from {}'.format(path))
    new_dataset = []
    data = pd.read_csv(path)
    nlp = English()
    spacy_tokenizer = nlp.tokenizer
    # df = pd.DataFrame(data, columns=['original_text', 'perturbed_text', 'perturbed_output', 'ground_truth_output'])
    for i in range(len(data)):
        if data['original_output'][i] != data['perturbed_output'][i]:
            new_perturbed_text = ' '.join(clean_str(data['perturbed_text'][i], tokenizer=spacy_tokenizer))
            new_dataset.append((new_perturbed_text, data['ground_truth_output'][i]))
    logger.info('{} set length: {}'.format('successful adversarial example data', len(new_dataset)))
    new_dataset = Dataset(new_dataset)
    return new_dataset


def load_ae_data_chatgpt(path):
    logger.info('load ae data from {}'.format(path))
    new_dataset = []
    chatgpt_result = pd.read_csv(path)
    data = pd.read_csv(path.replace('_chatgpt', ''))
    nlp = English()
    spacy_tokenizer = nlp.tokenizer
    id_in_chatgpt = 0
    # df = pd.DataFrame(data, columns=['original_text', 'perturbed_text', 'perturbed_output', 'ground_truth_output'])
    for i in range(len(data)):
        if data['result_type'][i] == 'Successful':
            if chatgpt_result['ChatGPT'][id_in_chatgpt] == "Yes":
                new_perturbed_text = ' '.join(clean_str(data['perturbed_text'][i], tokenizer=spacy_tokenizer))
                new_dataset.append((new_perturbed_text, data['ground_truth_output'][i]))
            id_in_chatgpt += 1
    logger.info('{} set length: {}'.format('successful adversarial example data', len(new_dataset)))
    logger.info('id_in_chatgpt: {}'.format(id_in_chatgpt))
    new_dataset = Dataset(new_dataset)
    return new_dataset, id_in_chatgpt


def load_data_module(data_loader, mode):
    new_dataset = []
    test_raw = data_loader.test_raw  # test -> train / eval
    test_pols = data_loader.test_pols
    test_raw, test_pols = shuffle_lists(test_raw, test_pols)
    for i in range(len(test_pols)):
        new_dataset.append((test_raw[i], test_pols[i]))
    return new_dataset


def set_param(dataset):
    if dataset == "imdb":
        max_len = 256
        bert_max_len = 256
        num_classes = 2
        val_split_size = 0.2
    elif dataset == "agnews":
        max_len = 128
        bert_max_len = 50
        num_classes = 4
        val_split_size = 0.05
    elif dataset == 'amazon':
        max_len = 256
        bert_max_len = 128
        num_classes = 2
        val_split_size = 0.1
    return max_len, bert_max_len, num_classes, val_split_size


def set_batch_size(use_BERT, dataset):
    batch_size_train = 100
    batch_size_val = 100
    batch_size_test = 100
    if use_BERT:
        batch_size_train = 32 if dataset == "agnews" else 16
        batch_size_val = 32 if dataset == "agnews" else 16
        batch_size_test = 32 if dataset == "agnews" else 16
    return batch_size_train, batch_size_val, batch_size_test


def set_lr(model_type):
    if 'lstm' in model_type:
        num_epoch = 50
        learning_rate = 1e-3
    if 'cnn' in model_type:
        num_epoch = 50
        learning_rate = 1e-3
    elif 'bert' in model_type:
        num_epoch = 10
        learning_rate = 3e-5
    return num_epoch, learning_rate


def log_model_state_dict(logger, model):
    """
    from utils: print_model_state_dict
    """
    logger.info("Model state dict:")
    for param_tensor in model.state_dict():
        logger.info("{}\t{}".format(param_tensor, model.state_dict()[param_tensor].size()))
    logger.info(
        "Num params: {}".format(sum([x.numel() for x in model.parameters() if x.requires_grad])))


# class CustomPytorchModelWrapper(ModelWrapper):
#     def __init__(self, model, tokenizer, params):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.config = params
#
#     def __call__(self, text_input_list):
#         all_input_ids = self.encode_fn(text_input_list)
#         dataset = TensorDataset(all_input_ids)
#         pred_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
#         # self.model.to(device)
#         self.model.eval()
#         prediction = []
#         for batch in pred_dataloader:
#             mask = (batch[0] > 0)
#             outputs = self.model(batch[0].cuda(), token_type_ids=None, attention_mask=mask.cuda())
#
#             logits = outputs[0]
#             logits = logits.detach().cpu().numpy()  # print(f'logits:{logits}')
#
#             prediction.append(logits)  # print(f'np.concatenate prediction:{np.concatenate(prediction, axis=0)}')
#
#         return np.concatenate(prediction, axis=0)
#
#     def encode_fn(self, text_list):
#         all_input_ids = []
#         for text in text_list:
#             input_ids = self.tokenizer.encode(
#                 text,
#                 max_length=self.config.bert_max_len,
#                 add_special_tokens=True,
#                 # pad_to_max_length=True,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#                 )
#             #     return_attention_mask=True,
#             all_input_ids.append(input_ids)
#         all_input_ids = torch.cat(all_input_ids, dim=0)
#         return all_input_ids
