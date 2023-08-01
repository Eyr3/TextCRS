import numpy as np
import secrets
from numbers import Real
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from noises.staircase import Staircase
from utils import load_pkl
import random
import os
import pickle
import sklearn
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import save_pkl
from numpy.linalg import norm
import json
from collections import defaultdict
import torch
import scipy.special


# f = open('/data/xinyu/results/fgws/attacks/chatgpt/decept_rate.csv', 'a+')
# # print("dataset\ttotal_ae\tdcpt_rate\tcosine_total\tcosine_success", file=f, flush=True)
# path = '/home/huangpeng/textRS/ChatGPT/data/bert_imdb/'
# filelist = os.listdir(path)
# for filename in filelist:
#     if 'successful' in filename:  #  and 'swap' in filename
#         total = []
#         data = pd.read_csv(path+filename)
#         ae_list = [i for i in range(len(data)) if (data['result_type'][i] == 'Successful' and data['ChatGPT Cosine Similarity'][i] < 1)]
#         ae_num = len(ae_list)
#         # ae_num = sum(int(data['result_type'][i] == 'Successful') for i in range(len(data)))
#         success_list = [i for i in range(len(data)) if (data['result_type'][i] == 'Successful' and
#                                                         data['ChatGPT'][i] == 'Yes' and data['ChatGPT Cosine Similarity'][i] < 1)]
#         cheat_num = len(success_list)
#         # for i in range(len(data['ChatGPT Cosine Similarity'])):
#         #     if data['ChatGPT Cosine Similarity'][i] > 1:
#         #         total.append(data['ChatGPT Cosine Similarity'][i]/100)
#         #     else:
#         cos_total = np.mean(data['ChatGPT Cosine Similarity'][ae_list])
#         cos_success = np.mean(data['ChatGPT Cosine Similarity'][success_list])
#         print('{}\t{}\t{}\t{}\t{}'.format(filename, ae_num, cheat_num/ae_num, cos_total, cos_success), file=f, flush=True)
# f.close()
#
# exit()

# model = '/data/xinyu/results/fgws/models/bert/imdb/noise1_k=50/best_model/pytorch_model.bin'
# data = torch.load(model)


# exit()

noise = 3
for model in ['newcnn',]:  # 'lstm', 'bert', 'cnn', 'newlstm', 'newbert', 'newcnn'
    # f = open('/data/xinyu/results/fgws/smooth/certify1/{}/acc_noise{}'.format(model, noise), 'a+')
    # print("dataset\tmodel\tacc", file=f, flush=True)
    for dataset in ['amazon', ]:  # 'agnews', 'amazon', 'imdb'  #_channel_1.0_mu
        for sigma in [0.3]:
            path = '/data/xinyu/results/fgws/smooth/certify1/{}/{}/noise{}/from0.1_noise_{}_sigma_{}_channel_1.0_mu_insert'.format(model, dataset, noise, sigma, sigma)
            data = pd.read_csv(path, '\t')
            # data = data[:500]
            acc = sum(data['correct']) / len(data)
            print("noise{}; dataset-{}; model-{}; sigma-{}; length-{}; acc-{}".format(noise, dataset, model, sigma, len(data), acc))
            # print('{}\t{}\t{}'.format(dataset, model, acc), file=f, flush=True)
    # f.close()


exit()
# def func1(max_len, z):
#     return (np.e*max_len/z)**z
#
# def func2(max_len, z, p):
#     return scipy.special.binom(max_len, z) * (p**z) * (1-p)**(max_len-z)
#
# x1 = np.linspace(1, 51, 50)
# y1 = func1(50, x1)
#
# plt.plot(x1, y1, 'r')
# plt.show()
#
# exit()
#
# channelpath = '/data/xinyu/results/fgws/models/newbert/amazon/channel_0.0noise3_g-n=0.1/best_model/pytorch_model.bin'
# channel = torch.load(channelpath)
#
# exit()
# Draw mean of each dimension in bert embedding vector
# # word_embeddings_file = '/data/xinyu/results/fgws/data/pretrained/transformers/paragram.npy'
# word_embeddings_file = '/data/xinyu/results/fgws/data/pretrained/paragramcf/paragram.npy'
# embedding_matrix = np.load(word_embeddings_file)
# word_select = np.random.randint(0, high=len(embedding_matrix), size=1000, dtype='l')
#
# emb_matrix = embedding_matrix[word_select]
# for i in range(0, 300, 30):
#     emb_dimension = emb_matrix[:, i]
#     plt.hist(emb_dimension, bins=40, color='red', histtype='stepfilled', alpha=0.75, )
#     plt.show()


# model = transformers.BertModel.from_pretrained("bert-base-uncased")
# tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
# word2index = np.load('/data/xinyu/results/fgws/data/pretrained/paragramcf/wordlist.pickle', allow_pickle=True)
#
# embedding_matrix = np.zeros((len(word2index), 768))
# np.save('/data/xinyu/results/fgws/data/pretrained/transformers/paragram.npy', embedding_matrix, allow_pickle=True)
#
# for word, index in word2index.items():
# # for i in range(1):  # len(wordlist)
#     inputs = tokenizer(word, return_tensors="pt")
#     outputs = model(**inputs)
#     word_vect = outputs.pooler_output.detach().numpy()
#     embedding_matrix[index] = word_vect
#
# np.save('/data/xinyu/results/fgws/data/pretrained/transformers/paragram.npy', embedding_matrix, allow_pickle=True)
#
# exit()


# mu_sigma = np.load('/data/xinyu/results/fgws/data/pretrained/gloVe/glove_data/mu_sigma_42B.npy')
# mu = list(mu_sigma[:, 0])
#
# with open(os.path.join('/data/xinyu/results/fgws/data/pretrained/gloVe/glove_data/', "config.json"), "w") as f:
#     json.dump(mu, f)

# Calculate embedding vector mean and std in each dimension
# # word_embeddings_file = '/data/xinyu/results/fgws/data/pretrained/gloVe/glove_data/glove.42B.300d.mat.npy'
# word_embeddings_file = '/data/xinyu/results/fgws/data/pretrained/transformers/paragram.npy'
# embedding_matrix = np.load(word_embeddings_file)
#
# # mu_sigma_42B = np.zeros((300, 2))
# mu_sigma_42B = np.zeros((768, 2))
#
# for i in range(0, len(embedding_matrix[0])):
#     # plt.hist(words_emb_matrix[:,10], bins=40, color='red', histtype='stepfilled', alpha=0.5, )
#     # plt.show()
#     mu_sigma_42B[i][0] = np.mean(embedding_matrix[:, i])
#     mu_sigma_42B[i][1] = np.std(embedding_matrix[:, i])
#
# # np.save('/data/xinyu/results/fgws/data/pretrained/gloVe/glove_data/mu_sigma_42B.npy', mu_sigma_42B)
# np.save('/data/xinyu/results/fgws/data/pretrained/transformers/mu_sigma.npy', mu_sigma_42B)
#
# exit()


# Display smallest top 10 l2 distance
# with open('/data/xinyu/results/fgws/data/pretrained/paragramcf/mse_dist.p', "rb") as f:
#     mse_dist_mat = pickle.load(f)
#
# with open('/data/xinyu/results/fgws/data/pretrained/transformers/mse_dist.p', 'rb') as f:
#     mse_dist_mat = pickle.load(f)
#
# # nn_matrix = np.load('/data/xinyu/results/fgws/data/pretrained/paragramcf/nn.npy')
# l = []
#
# for i in range(0, len(mse_dist_mat), 1000):
#     # l2 = [np.sqrt(mse_dist_mat[i][j]) for j in nn_matrix[i][1:]]
#     l2 = [np.sqrt(mse_dist_mat[i][index]) for index, mse in mse_dist_mat[i].items()]
#     if len(l2) < 11:
#         continue
#     else:
#         l.append(np.sort(l2)[1:11])
#
# f.close()
#
# l = np.array(l).flatten()
#
# plt.hist(l, bins=40, color='red', histtype='stepfilled', alpha=0.75)
# plt.show()
#
# exit()


# Calculate MES distance between synonym word, i.e., l2distance**2
word2index = np.load('/data/xinyu/results/fgws/data/pretrained/paragramcf/wordlist.pickle', allow_pickle=True)

# LSTM
word_embeddings_file = '/data/xinyu/results/fgws/data/pretrained/paragramcf/paragram.npy'
embedding_matrix = torch.from_numpy(np.load(word_embeddings_file))  #.to('cuda:5')

nn_matrix = np.load('/data/xinyu/results/fgws/data/pretrained/paragramcf/nn.npy')  # total N words with their K synonym

mse_dist_mat = defaultdict(dict)  # np.zeros((len(embedding_matrix), 100))

for word, index in word2index.items():
    e1 = embedding_matrix[index]
    for j in nn_matrix[index]:
        mse_dist_mat[index][j] = torch.sum((e1 - embedding_matrix[j]) ** 2).item()

with open('/data/xinyu/results/fgws/data/pretrained/paragramcf/self_mse_dist.p', 'wb') as handle:
    pickle.dump(mse_dist_mat, handle)

exit()

# BERT
word_embeddings_file = '/data/xinyu/results/fgws/data/pretrained/transformers/paragram.npy'
embedding_matrix = torch.from_numpy(np.load(word_embeddings_file)).to('cuda:0')

nn_matrix = np.load('/data/xinyu/results/fgws/data/pretrained/paragramcf/nn.npy')  # total N words with their K synonym

mse_dist_mat = defaultdict(dict)  # np.zeros((len(embedding_matrix), 100))

for word, index in word2index.items():
    e1 = embedding_matrix[index]
    for j in nn_matrix[index]:
        mse_dist_mat[index][j] = torch.sum((e1 - embedding_matrix[j]) ** 2).item()

with open('/data/xinyu/results/fgws/data/pretrained/transformers/mse_dist.p', 'wb') as handle:
    pickle.dump(mse_dist_mat, handle)

exit()
#     e2 = torch.tensor(e2).to(utils.device)
#     mse_dist = torch.sum((e1 - e2) ** 2).item()
#     self._mse_dist_mat[a][b] = mse_dist



# syn_path = '/data/xinyu/results/fgws/data/pretrained/paragramcf/'
# word2index = np.load('{}wordlist.pickle'.format(syn_path), allow_pickle=True)
# index2word = {}
# for word, index in word2index.items():
#     index2word[index] = word
# np.save('{}index2word_default.pickle'.format(syn_path), index2word, allow_pickle=True)


exit()

# Generate Synonym with TextAttack (paragramcf)
# dist_mat = sklearn.metrics.pairwise.cosine_distances(embedding)
#
# print("Dist mat shape {}".format(np.shape(dist_mat)))
#
# nn_matrix = [[] for i in range(len(dist_mat))]
# for idx in range(len(dist_mat)):
#     neighbors = dist_mat[idx]
#     sorted_idx = np.argsort(neighbors).tolist()
#     nn_matrix[idx] = sorted_idx[:250]
# np.save('/data/xinyu/results/fgws/data/pretrained/paragramcf/syn_250/nn_matrix.npy', nn_matrix, allow_pickle=True)

exit()


# show and compare the size of mat
# wordsList1 = np.load('/home/zhangxinyu/.cache/textattacknew/word_embeddings/glove200/glove.wordlist.npy')
# wordsList1 = wordsList1.tolist()  # Originally loaded as numpy array
# wordsList2 = np.load('/data/xinyu/results/fgws/data/pretrained/gloVe/glove_data/glove.6B.300d.wordlist.npy')
# wordsList2 = wordsList2.tolist()
# wordVectors1 = np.load('/home/zhangxinyu/.cache/textattacknew/word_embeddings/glove200/glove.6B.200d.mat.npy')
# print(wordVectors1.shape)
# wordVectors2 = np.load('/data/xinyu/results/fgws/data/pretrained/gloVe/glove_data/glove.42B.300d.mat.npy')
# print(wordVectors2.shape)


# Glove -> Wordlist + mat used in TextAttack ref: https://blog.csdn.net/keeppractice/article/details/108473693
# embeddings_dict = {}
# with open("/data/xinyu/results/fgws/data/pretrained/gloVe/glove.42B.300d.txt", 'r', encoding="utf-8") as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:], "float32")
#         embeddings_dict[word] = vector
# np.save('/data/xinyu/results/fgws/data/pretrained/gloVe/glove_data/glove.42B.300d.wordlist.npy', np.array(list(embeddings_dict.keys())))
# np.save('/data/xinyu/results/fgws/data/pretrained/gloVe/glove_data/glove.42B.300d.mat.npy', np.array(list(embeddings_dict.values()), dtype='float32'))


exit()

# Generate neighbors' distance matrix from the original code fgws-main
# dist_mat = np.load("/home/zhangxinyu/code/fgws-main/data/imdb/dist_mat.npy", allow_pickle=True)
# neighbors_idx = [[] for i in range(len(dist_mat))]
# for idx in range(len(dist_mat)):
#     neighbors = dist_mat[idx]
#     sorted_idx = np.argsort(neighbors).tolist()
#     # sorted_idx.remove(idx)
#     neighbors_idx[idx] = sorted_idx[:250]
# np.save("/data/xinyu/results/fgws/data/data/imdb/dist_mat_neighbor.npy", neighbors_idx, allow_pickle=True)


exit()
# Staircase Mechanism
# staircase_mech = Staircase(epsilon=1, gamma=1, sensitivity=1, random_state=1)
#
# data = np.zeros(1000)
# for i in range(1000):
#     data[i] = staircase_mech.randomise(0)
#
# plt.title("epsilon=1, gamma=1, sensitivity=1, random_state=1")
# weights = np.ones_like(data)/float(len(data))
# plt.hist(data, bins=40, color='red', histtype='stepfilled', alpha=0.75, weights=weights)
# plt.show()


exit()

a = [[1, 2, 3], [3, 4, 3], [5, 6, 3]]
b = [[5, 6, 3], [3, 4, 3], [1, 2, 3]]

# aa = np.dot(a, np.linalg.inv(a))
# print(aa)

# a = [1,2,3]
x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
x1 = np.array([[1, 0, 0], [0.2, 0.3, 0.5], [0, 0, 1]])

print(np.dot(np.dot(x, x1), a))
print(np.dot(x, np.dot(x1, a)))
exit()

xr = x.ravel()
x1r = x1.ravel()
cosine = np.dot(xr, x1r)/(norm(xr)*norm(x1r))
print(cosine)
# bb = np.dot(x,a)
# print(bb)

# A=[[8,1,6],[3,5,7],[4,9,2]]
# B=[[4,9,2],[3,5,7],[8,1,6]]
# bb=np.dot(B, np.linalg.inv(A))
# print(bb)
