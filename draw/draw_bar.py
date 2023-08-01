import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib
# import palettable
# sns.set()
sns.set_style('darkgrid')
# font_size=23
font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

# plt.rc('font', size=font_size) #controls default text size
# plt.rc('axes', titlesize=font_size) #fontsize of the title
# plt.rc('axes', labelsize=font_size) #fontsize of the x and y labels
# plt.rc('xtick', labelsize=17) #fontsize of the x tick labels
# plt.rc('ytick', labelsize=font_size) #fontsize of the y tick labels
# plt.rc('legend', fontsize=font_size) #fontsize of the legend

# patch='//'

# ea = pd.read_csv('result/evaulation_tar.CSV')
# name = 'target_evaul_noniid_fpr'#robust

# if 'tpr' in name:
#     dnc_cl = pd.DataFrame(ea, columns=['Classical DnC TPR']).values.ravel().tolist()
#     our_cl = pd.DataFrame(ea, columns=['Classical Our TPR']).values.ravel().tolist()
#     dnc_ea = pd.DataFrame(ea, columns=['Evasion DnC TPR']).values.ravel().tolist()
#     our_ea = pd.DataFrame(ea, columns=['Evasion Our TPR']).values.ravel().tolist()
#     labels=['(Classical attacks) DnC','(Evasion attacks) DnC','(Classical attacks) Ours','(Evasion attacks) Ours','TPR (%)']
#     colors=['#fc8d59','#4393c3']
# else:
#     our_ea = pd.DataFrame(ea, columns=['Evasion Our FPR']).values.ravel().tolist()
#     our_cl = pd.DataFrame(ea, columns=['Classical Our FPR']).values.ravel().tolist()
#     dnc_cl = pd.DataFrame(ea, columns=['Classical DnC FPR']).values.ravel().tolist()
#     dnc_ea = pd.DataFrame(ea, columns=['Evasion DnC FPR']).values.ravel().tolist()
#     labels=['(Classical attacks) DnC','(Evasion attacks) DnC','(Classical attacks) Ours','(Evasion attacks) Ours','FPR (%)']
#     colors=['#cb181d','#2ca25f']

# if 'untarget' in name:
#     att_type = ['C-A', 'C-B', 'C-C',
#                 'E-A', 'E-B', 'E-C', 'E-D', 'E-E', 'E-F', 'E-G', 'E-H']
#     classical = 3
#     evasion = 8
#     if 'noniid' in name:
#         D_M = ['EMNIST (SimpleNet)',  'CIFAR10 (AlexNet)','CIFAR10 (ResNet18)']
#     else:
#         D_M = ['MNIST (SimpleNet)', 'CIFAR10 (AlexNet)', 'CIFAR10 (ResNet18)']
# else:
#     att_type = ['BN\nS', 'BN\nM', 'BN\nBM', 'BL\nS', 'BL\nM', 'BL\nBM', 'DB\nS', 'DB\nM',
#             'BN\nS', 'BN\nM', 'BN\nBM', 'BL\nS', 'BL\nM', 'BL\nBM', 'DB\nS', 'DB\nM']
#     har_att_type = ['BN\nS', 'BN\nM', 'BN\nBM',  'DB\nS', 'DB\nM',
#                  'BL\nS', 'BL\nM', 'BL\nBM', 'DB\nS', 'DB\nM']
#     classical = evasion=8
#     har_classical = har_evasion = 5
#     D_M = ['CIFAR10 (VGG16)','CIFAR10 (ResNet18)',  'GTSRB (ResNet34)', 'HAR (DNN)']
# att_type = ['BN(A-S)', 'BN(A-M)', 'BN(A-BM)', 'BL(A-S)', 'BL(A-M)', 'BL(A-BM)', 'DB(A-S)', 'DB(A-M)']
# x = np.arange(len(att_type))

Accs = \
{'ag_lstm': \
[90.12, 	89.22, 	91.20, 	91.60, 
89.59, 	88.51, 	90.60, 	89.60, 
88.82, 	87.83, 	89.40, 	86.80, ],

'ag_bert': \
[93.24, 	92.74, 	95.20, 	93.20, 
92.75,	91.96, 	93.40, 	88.20, 
92.63, 	90.99, 	92.40, 	88.80, ],

'amazon_lstm': \
[87.86, 	86.40, 	83.20, 	85.60, 
87.29, 	85.62, 	82.60, 	80.40, 
86.23, 	84.19, 	84.40, 	80.80, ],

'amazon_bert': \
[93.91, 	92.28, 	93.00, 	87.40, 
93.07, 	90.73, 	90.60, 	84.20, 
91.62, 	89.39, 	88.20, 	82.40, ],

'imdb_lstm': \
[83.39, 	81.90, 	83.40, 	80.20, 
82.58, 	80.52, 	86.60, 	79.40, 
81.07, 	79.06, 	80.80, 	78.20, ],

'imdb_bert': \
[91.52, 	89.22, 	87.00, 	83.40, 
90.42, 	88.07, 	84.20, 	81.00, 
88.68, 	83.73, 	76.60, 	72.40, ]
}

fig, ax = plt.subplots(constrained_layout=True)
fig.set_size_inches(9, 5.5)

name = 'imdb_bert'
data = np.array(Accs[name]).reshape((3,4))
ylim_range={'ag_lstm':[86, 94],  'ag_bert':[87, 98],
            'amazon_lstm':[79, 92],  'amazon_bert':[80, 99],
            'imdb_lstm':[77, 91],  'imdb_bert':[70, 99],
            }

X = ['$s$=50','$s$=100','$s$=250'] 
bar_width = 0.25
opacity = 1
offset = bar_width/2

acc_ours = data[:, 0]
acc_safer = data[:, 1]

cer_acc_ours = data[:, 2]
cer_acc_safer = data[:, 3]

X_axis = np.arange(len(X))

plt.bar(X_axis - offset, acc_safer, bar_width, alpha=opacity, color='#4c72b0', label = 'Benign Acc._SAFER')
plt.bar(X_axis + offset, acc_ours, bar_width, alpha=opacity, color='#dd8452', label = 'Benign Acc._Ours')

plt.bar(len(X) + X_axis - offset, cer_acc_safer, bar_width, alpha=opacity, color='#55a868', label = 'Cert. Acc._SAFER')
plt.bar(len(X) + X_axis + offset, cer_acc_ours, bar_width, alpha=opacity, color='#c44e52', label = 'Cert. Acc._Ours')

ax.set_ylim(ylim_range[name][0], ylim_range[name][1])
ax.set_xticks(np.arange(2*len(X)), X*2)
ax.set_xlabel("Smoothing Paramter (Size of Thesaurus)")
ax.set_ylabel("Benign / Cert. Acc. (%)")
# ax.set_title("AG_LSTM")
ax.legend(loc='upper right', ncol=2)
plt.show()

plt.tight_layout()
plt.savefig(f'/home/zhangxinyu/code/fgws-main/draw/compare_safer/ours_safer_{name}.png', bbox_inches="tight") #

exit()

# fig = plt.figure(figsize=(35, 5), tight_layout=True)  # cl:12,3; ev: 15,6
# gs = gridspec.GridSpec(4, 29)  # 分为3行3列

# x = np.arange(len(att_type))  # the label locations
width = 0.5 / 2  # the width of the bars

for i,D_M_n in enumerate(D_M):

    if 'HAR' not in D_M_n:
        ax1 = plt.subplot(gs[:3, i * 8:(i+1)*8])
        ax1.set_prop_cycle('color', palettable.colorbrewer.qualitative.Set1_9.mpl_colors)
        plt.bar(x[:classical] - width / 2,
                dnc_cl[i * classical:(i + 1) * classical] ,
                width, label=labels[0],color=colors[0])  #
        plt.bar(x[classical:] - width / 2,
                dnc_ea[i * evasion:(i + 1) * evasion],
                width, label=labels[1], color=colors[0], hatch='//')
        plt.bar(x[:classical] + width / 2,
                our_cl[i * classical:(i + 1) * classical],
                width, label=labels[2],color=colors[1])
        plt.bar(x[classical:] + width / 2,
                our_ea[i * evasion:(i + 1) * evasion],
                width, label=labels[3], color=colors[1],  hatch='//')
        plt.xticks(x, att_type, wrap=True)
    else:
        ax1 = plt.subplot(gs[:3, i * 8:])
        ax1.set_prop_cycle('color', palettable.colorbrewer.qualitative.Set1_9.mpl_colors)
        plt.bar(x[:har_classical] - width / 2,
                dnc_cl[i * classical:i * classical+har_classical],
                width, label=labels[0], color=colors[0])  #
        plt.bar(x[har_classical:har_classical+har_evasion] - width / 2,
                dnc_ea[i * evasion:i * evasion+har_evasion],
                width, label=labels[1], color=colors[0], hatch='//')
        plt.bar(x[:har_classical] + width / 2,
                our_cl[i * classical:i * classical+har_classical],
                width, label=labels[2], color=colors[1])
        plt.bar(x[har_classical:har_classical+har_evasion] + width / 2,
                our_ea[i * evasion:i * evasion+har_evasion],
                width, label=labels[3], color=colors[1], hatch='//')
        plt.xticks(x[:har_classical+har_evasion], har_att_type, wrap=True)
    #
    plt.title(D_M_n)

    # plt.legend(loc=4)
    if i==0:
        plt.ylabel(labels[-1])

ax1=plt.subplot(gs[3,0])
ax1.clear()  # clears the random data I plotted previously
ax1.set_axis_off()  # removes the XY axes
#
lines, labels = fig.axes[0].get_legend_handles_labels()
# # Add legend to bottom-right ax
# # plt.legend(lines, labels, loc='upper center', ncol=3)
#
fig.legend(lines, labels, ncol=4,bbox_to_anchor=(0.52,0.19),loc='upper center')
if 'untar' not in name:
    fig.text(0.035, 0.24, 'Trigger\nInject', fontsize=17,horizontalalignment='center',verticalalignment='center')
# fig.text(0.67, 0.252, 'Trigger\nInject', fontsize=17,horizontalalignment='center',verticalalignment='center')
# fig.text(0.985, 0.252, 'Trigger\nInject', fontsize=17,horizontalalignment='center',verticalalignment='center')
# fig.legend(loc='upper center', bbox_to_anchor=(0.52, 0.17), fancybox=True, ncol=3)
plt.tight_layout()
plt.savefig(f'{name}.png', bbox_inches="tight") #

# plt.show()
