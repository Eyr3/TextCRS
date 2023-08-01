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
font = {'family' : 'Times New Roman',
        # 'weight' : 'bold',
        'size'   : 23}
matplotlib.rc('font', **font)


# Synonym Substitution			Word Reordering			Word Insertion			Word Deletion		
Accs = \
{'lstm': \
[86.4,	0.0,	88.8,	0.0,	0.0,	92.3,	0.0,	0.0,	88.6,	0.0,	0.0,	91.1,],
'bert': \
[92.1,	85.6,	92.7,	0.0,	0.0,	95.5,	0.0,	0.0,	93.5,	0.0,	0.0,	94.5,],
}


fig, ax = plt.subplots(constrained_layout=True)
fig.set_size_inches(11.5, 5)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.weight'] = 'bold'

name = 'ag_bert' # _certify _benign: clean
# if 'lstm' in name:
#     data = np.array(Accs['lstm']).reshape((9,4))/100
# else:
#     data = np.array(Accs['bert']).reshape((9,4))/100
ylim_range={'ag_lstm':[0, 1.0],  'lstm_certify':[0.5, 0.9],
            'ag_bert':[0, 1.0],  'bert_certify':[0.6, 0.95],
            }

if 'lstm' in name:
    data = np.array(Accs['lstm']).reshape((4,3))/100
else:
    data = np.array(Accs['bert']).reshape((4,3))/100


# X = ['$\sigma$=0.1','$\sigma$=0.2','$\sigma$=0.3', '$\sigma$=0.1','$\sigma$=0.2','$\sigma$=0.3', '$\sigma$=0.1','$\sigma$=0.2','$\sigma$=0.3'] 
X = ['Synonym\n Substitution','Word\n Reordering','Word\n Insertion','Word\n Deletion'] 

bar_width = 0.3
opacity = 1
offset = bar_width/2

X_axis = np.arange(len(X))

# cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# if 'benign' in name:
#     # plt.bar(X_axis - offset, acc_wo, bar_width, hatch='//', edgecolor='#1f77b4', color='white', alpha=opacity, label='Benign Acc._w/o enhance')
#     # plt.bar(X_axis + offset, acc_w, bar_width, hatch='+', edgecolor='#ff7f0e', color='white', alpha=opacity, label='Benign Acc._w/ enhance')
#     plt.bar(X_axis - offset, acc_wo, bar_width, color='#1f77b4', alpha=0.8, label='w/o enhance') # Clean Acc._
#     plt.bar(X_axis + offset, acc_w, bar_width, color='#e19153', alpha=opacity, label='w/ enhance') # Clean Acc._
# else:
# plt.bar(X_axis - offset, cer_acc_wo, bar_width, hatch='//', edgecolor='#2ca02c', color='white', alpha=opacity, label='Cert. Acc._w/o enhance')
# plt.bar(X_axis + offset, cer_acc_w, bar_width, hatch='+', edgecolor='#d62728', color='white', alpha=opacity, label='Cert. Acc._w/ enhance')


plt.bar(X_axis - 2* offset, data[:, 0].flatten(), bar_width, color='#478ec0', alpha=opacity, label='SAFER') # Cert. Acc._
plt.bar(X_axis , data[:, 1], bar_width, color='#48b2a3', alpha=opacity, label='CISS') # Cert. Acc._
plt.bar(X_axis + 2* offset, data[:, 2], bar_width, color='#da6046', alpha=opacity, label='Ours') # Cert. Acc._


# plt.axvline(2.5, linestyle='-', linewidth=5, c='white')
# plt.axvline(5.5, linestyle='-', linewidth=5, c='white')

# plt.axvline(2.5, linestyle=':', linewidth=3, c='gray')
# plt.axvline(5.5, linestyle=':', linewidth=3, c='gray')

ax.set_ylim(0, 1)
ax.set_xticks(np.arange(len(X)), X, weight='bold')

# label = ax.set_xlabel('$\sigma$', weight='bold')
ax.xaxis.set_label_coords(1.0, -0.025)
# ax.set_xlabel("Smoothing Paramter (Gaussian Noise Level)")
ax.set_ylabel("Certified accuracy", weight='bold')

if 'ag_lstm' in name:
    ax.set_title("AG's News, LSTM", fontsize=23, weight='bold')
elif 'ag_bert' in name:
    ax.set_title("AG's News, BERT", fontsize=23, weight='bold')
elif 'amazon_lstm' in name:
    ax.set_title("Amzon, LSTM", fontsize=23, weight='bold')
elif 'amazon_bert' in name:
    ax.set_title("Amzon, BERT", fontsize=23, weight='bold')
elif 'imdb_lstm' in name:
    ax.set_title("IMDB, LSTM", fontsize=23, weight='bold')
elif 'imdb_bert' in name:
    ax.set_title("IMDB, BERT", fontsize=23, weight='bold')


ax.legend(loc='lower right', ncol=3, fontsize=23) # 
# else:
#     ax.legend(loc='upper right', ncol=1) # , fontsize=21
plt.show()

plt.tight_layout()
plt.savefig(f'/home/zhangxinyu/code/fgws-main/draw/ppt/{name}.png', bbox_inches="tight")  #, format="pdf"

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

