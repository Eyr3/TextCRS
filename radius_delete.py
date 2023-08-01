import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import pandas as pd
from helper import set_param
import scipy.special
import math


def func(n, z1, p):
    return scipy.special.binom(n, z1) * (p**z1) * (1-p)**(n-z1)


def func2(n, z1, delta, p):
    return scipy.special.binom(z1, delta) * scipy.special.binom(n, z1) * (p**z1) * (1-p)**(n-z1)


def func_CDF1(z_center, z_1, n):  # B: (z_1, n-z_1+1)
    x = np.arange(z_center-z_1-1, z_center-z_1)
    y = func(n, x, sigma)
    return sum(y)


def func2_CDF1(z_center, z_1, n, delta):  # B: (z_1, n-z_1+1)
    x = np.arange(z_center-z_1-1, z_center-z_1)
    y = func2(n, x, delta, sigma)
    return sum(y)


def radius_z(z_start, z_stop, z_step, max_len, pBBar, th):
    radius = 0
    last_z = -1
    for z in range(z_start, z_stop, z_step):
        if func(max_len, z, sigma) < pBBar:
            last_z = z
            break
    for delta in range(int(last_z / 2), 0, -1):  # int(max_len / 2)
        # if (e*max_len/delta)**delta <= th:
        # if scipy.special.binom(max_len, delta) <= th:
        if scipy.special.binom(last_z, delta) <= th:
            # if delta <= math.log(th, 2*np.e):
            radius = delta
            break
    return radius


def write_radius():
    data = pd.read_csv(path, '\t')
    data['radius'] = ''

    if 'lstm' in model:
        max_len, _, _, _ = set_param(dataset)
    else:
        _, max_len, _, _ = set_param(dataset)

    f = open('{}_radius'.format(path), 'w')
    print("idx\tlabel\tpredict\tpABar\tpBBar\tcorrect\tradius", file=f, flush=True)

    for i in range(len(data)):
        radius = 0
        pABar = data['pABar'][i]
        if dataset == 'agnews':
            pBBar = min(1-pABar, data['pBBar'][i])
        else:
            pBBar = 1 - pABar

        if data['correct'][i]:
            th = pABar / pBBar

            z_center = int(sigma*max_len)
            z_start, z_stop, z_step = z_center, max_len, 1
            radius1 = radius_z(z_start, z_stop, z_step, max_len, pBBar, th)
            z_start, z_stop, z_step = z_center, 0, -1
            radius2 = radius_z(z_start, z_stop, z_step, max_len, pBBar, th)
            radius = max(radius1, radius2)
            # # f_range = func(max_len, np.arange(z_min, z_max), sigma)

            # last_z = -1
            # for z in range(z_start, z_stop, z_step):
            #     if func(max_len, z, sigma) < pBBar:
            #         last_z = z
            #         break
            #
            # for delta in range(int(max_len / 2), 0, -1):  # int(last_z/2)
            #     # if (e*max_len/delta)**delta <= th:
            #     if scipy.special.binom(max_len, delta) <= th:
            #     # if scipy.special.binom(last_z, delta) <= th:
            #     # if delta <= math.log(th, 2*np.e):
            #         radius = delta
            #         break

        print("{}\t{}\t{}\t{:.5}\t{:.5}\t{}\t{}".format(  # \t{}
            data['idx'][i], data['label'][i], data['predict'][i], pABar, pBBar,
            data['correct'][i], radius), file=f, flush=True)

    f.close()


if __name__ == '__main__':
    # model = 'bert'
    # dataset = 'agnews'
    # sigma = 0.5

    for model in ['cnn']:  # ,'lstm'
        for dataset in ['amazon', 'imdb']:  # 'amazon', 'imdb'
            for sigma in [0.3, 0.5, 0.7]:
                path = '/data/xinyu/results/fgws/smooth/certify1/{}/{}/noise4/noise_{}_sigma_{}'.\
                    format(model, dataset, sigma, sigma)
                write_radius()
