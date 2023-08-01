import numpy as np
import pickle
import matplotlib.pyplot as plt
import transformers


# https://huggingface.co/spaces/anonymous8/RPD-Demo/resolve/49437524d52c48d8ddbdaf8dd2edfaec99e6718d/textattack/shared/word_embeddings.py

model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
wordlist = np.load('/data/xinyu/results/fgws/data/pretrained/paragramcf/wordlist.pickle', allow_pickle=True)

for i in range(1):  # len(wordlist)
    inputs = tokenizer(wordlist[i], return_tensors="pt")
    outputs = model(**inputs)
    word_vect = outputs.pooler_output.detach().numpy()

exit()

# embedding = np.load('/data/xinyu/results/fgws/data/pretrained/paragramcf/paragram.npy')
# nn_matrix = np.load('/data/xinyu/results/fgws/data/pretrained/paragramcf/nn.npy', allow_pickle=True)
# cos_sim_file = '/data/xinyu/results/fgws/data/pretrained/paragramcf/cos_sim.p'
mse_dist_file = '/data/xinyu/results/fgws/data/pretrained/paragramcf/mse_dist.p'  # l2 distance
with open(mse_dist_file, "rb") as f:
    mse_dist_mat = pickle.load(f)

word_select = np.random.randint(0, high=len(mse_dist_mat), size=1000, dtype='l')

small_dist = []
radius = 2  # 0.5
for i in word_select:
    mse_d = list(mse_dist_mat[i].values())[1:]
    small_dist += [d for d in mse_d if d < radius]

print(len(small_dist))
plt.hist(small_dist, bins=40, color='red', histtype='stepfilled', alpha=0.75,)
plt.show()


# bert, calculate embedding l2 distance
# e1 = self.embedding_matrix[a]
# e2 = self.embedding_matrix[b]
# e1 = torch.tensor(e1).to(utils.device)
# e2 = torch.tensor(e2).to(utils.device)
# mse_dist = torch.sum((e1 - e2) ** 2).item()
# self._mse_dist_mat[a][b] = mse_dist
