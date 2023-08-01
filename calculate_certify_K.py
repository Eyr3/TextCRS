import pandas as pd
import sys
import numpy as np
import re

from textattacknew.certify_K import certify_K

model_type = 'bert'
beta = 0.3

fn = 'agnews'
global_d = 50 if model_type == 'bert' else 128

# fn = 'amazon'
# global_d = 128 if model_type == 'bert' else 256
#
# fn = 'imdb'
# global_d = 256

# file = sys.argv[1]
file = '/data/xinyu/results/fgws/smooth/certify1/{}/{}/noise4/noise_{}_sigma_{}'.format(model_type, fn, beta, beta)
df = pd.read_table(file, sep="\t")

label = df["label"]
df.loc[:, "label"] = [int(re.findall(r"\d+", label[i])[0]) for i in range(len(label))]
label = df["label"]
# label.values = [int(re.findall(r"\d+", label[i])[0]) for i in range(len(label))]
predict = df["predict"]
pAHat = df["pABar"]
accurate = df["correct"]
#print(pAHat)

test_num = predict == label
#print(sum(test_num))
test_acc = sum(test_num) / float(len(label))
#print(sum(df["correct"]))
print('certify acc:', sum(accurate)/len(label))

# alpha = float(sys.argv[2])
alpha = 0.001
print('alpha = {}'.format(alpha))

K = np.zeros(int(0.9*global_d), dtype=int)
for idx in range(len(pAHat)):
	if accurate[idx]:
		v = certify_K(pAHat[idx], alpha, global_d)
		print('pAHat:', pAHat[idx], 'Certified K:', v)
		K[v] += 1
print(K)

K_cer = np.cumsum(K[::-1])[::-1]

for idx in range(len(K_cer)):
	print(idx+1, K_cer[idx])


# fp = open('../thresholds/{}/{}_bi.txt'.format(fn, alpha), 'w')
# for idx in range(len(K_cer)):
# 	print(idx+1, K_cer[idx])
# 	fp.write('{}\t{}\n'.format(idx+1, K_cer[idx]))
# fp.close()

# ##Core: P=0.7  alpha=0.99, N=10K
# (1, 60)
# (2, 60)
# (3, 60)
# (4, 59)
# (5, 59)
# (6, 57)
# (7, 55)
# (8, 55)
# (9, 53)
# (10, 51)
# ...
