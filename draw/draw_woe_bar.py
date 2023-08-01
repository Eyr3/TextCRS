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


#acc_wo, acc_w, cer_acc_wo, cer_acc_w
Accs = \
{'lstm': \
[89.37,	90.38,	81.8,	84.2,   # 'agnews_lstm'
85.39,	86.66,	79,	82.4,
84.45,	84.83,	73.8,	78.2, 
86.67,	88.51,	57.6,	71.8,       #],'amazon_lstm': \ [
73.17,	83.61,	51.8,	61.4,
64.05,	79.53,	55.6,	58.8,   
82.02,	84.86,	70,	69.6,   #],'imdb_lstm': \ [
74.7,	80.45,	61.2,	64.2,
73.03,	77.96,	62,	58.4, ],

# {'lstm': \
# [89.37,	90.38,	82.6,	84.8,   # 'agnews_lstm'
# 85.39,	86.66,	80.2,	84,
# 84.45,	84.83,	76.4,	80.4, 
# 86.67,	88.51,	59,	74.2,       #],'amazon_lstm': \ [
# 73.17,	83.61,	56,	65.8,
# 64.05,	79.53,	61.4,	62.4,   
# 82.02,	84.86,	70.6,	69.6,   #],'imdb_lstm': \ [
# 74.7,	80.45,	64.4,	68.6,
# 73.03,	77.96,	63,	61.6, ],

'bert': \
[93.43,	89.7,   75.4,   73.2,
93.05,	88.39,	84.2,   70.8,
91.78,  87.21,   80.6,   69,
94.64,	89.7,	80.6,	67.8,#],'amazon_bert': \ [
94.43,	86.3,	78.4,	66.6, 
92.89,   84.25,   72.8,   68.2,
91.88,	82.6,	82.6,	74.2, #],'imdb_bert': \[
91.68,	83.35,	87,	74.2, 
87.49,   79.39,   75, 69.6,
],
}

fig, ax = plt.subplots(constrained_layout=True)
fig.set_size_inches(9, 4.5)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.weight'] = 'bold'

name = 'lstm_benign' # _certify _benign: clean
if 'lstm' in name:
    data = np.array(Accs['lstm']).reshape((9,4))/100
else:
    data = np.array(Accs['bert']).reshape((9,4))/100
ylim_range={'lstm_benign':[0.6, 1.0],  'lstm_certify':[0.5, 0.9],
            'bert_benign':[0.78, 1.0],  'bert_certify':[0.6, 0.95],
            }

if 'lstm' in name:
    # X = ['$\sigma$=0.1','$\sigma$=0.2','$\sigma$=0.3', '$\sigma$=0.1','$\sigma$=0.2','$\sigma$=0.3', '$\sigma$=0.1','$\sigma$=0.2','$\sigma$=0.3'] 
    X = ['0.1','0.2','0.3', '0.1','0.2','0.3', '0.1','0.2','0.3'] 
else:
    X = ['$\sigma$=0.5','$\sigma$=1.0','$\sigma$=1.5', '$\sigma$=0.5','$\sigma$=1.0','$\sigma$=1.5', '$\sigma$=0.5','$\sigma$=1.0','$\sigma$=1.5'] 

bar_width = 0.3
opacity = 1
offset = bar_width/2

acc_wo = data[:, 0]
acc_w = data[:, 1]

cer_acc_wo = data[:, 2]
cer_acc_w = data[:, 3]

X_axis = np.arange(len(X))

# cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

if 'benign' in name:
    # plt.bar(X_axis - offset, acc_wo, bar_width, hatch='//', edgecolor='#1f77b4', color='white', alpha=opacity, label='Benign Acc._w/o enhance')
    # plt.bar(X_axis + offset, acc_w, bar_width, hatch='+', edgecolor='#ff7f0e', color='white', alpha=opacity, label='Benign Acc._w/ enhance')
    plt.bar(X_axis - offset, acc_wo, bar_width, color='#1f77b4', alpha=0.8, label='w/o enhance') # Clean Acc._
    plt.bar(X_axis + offset, acc_w, bar_width, color='#e19153', alpha=opacity, label='w/ enhance') # Clean Acc._

else:
    # plt.bar(X_axis - offset, cer_acc_wo, bar_width, hatch='//', edgecolor='#2ca02c', color='white', alpha=opacity, label='Cert. Acc._w/o enhance')
    # plt.bar(X_axis + offset, cer_acc_w, bar_width, hatch='+', edgecolor='#d62728', color='white', alpha=opacity, label='Cert. Acc._w/ enhance')
    plt.bar(X_axis - offset, cer_acc_wo, bar_width, color='#48b2a3', alpha=opacity, label='w/o enhance') # Cert. Acc._
    plt.bar(X_axis + offset, cer_acc_w, bar_width, color='#da6046', alpha=opacity, label='w/ enhance') # Cert. Acc._

plt.axvline(2.5, linestyle='-', linewidth=5, c='white')
plt.axvline(5.5, linestyle='-', linewidth=5, c='white')

# plt.axvline(2.5, linestyle=':', linewidth=3, c='gray')
# plt.axvline(5.5, linestyle=':', linewidth=3, c='gray')

ax.set_ylim(ylim_range[name][0], ylim_range[name][1])
ax.set_xticks(np.arange(len(X)), X, weight='bold')

label = ax.set_xlabel('$\sigma$', weight='bold')
ax.xaxis.set_label_coords(1.0, -0.025)
# ax.set_xlabel("Smoothing Paramter (Gaussian Noise Level)")
if 'benign' in name:
    ax.set_ylabel("Clean accuracy", weight='bold')
else:    
    ax.set_ylabel("Certified accuracy", weight='bold')
ax.set_title("AG               Amazon             IMDB", fontsize=23, weight='bold')
if 'lstm' in name:
    ax.legend(loc='upper right', ncol=1, fontsize=23) # 
else:
    ax.legend(loc='upper right', ncol=1) # , fontsize=21
plt.show()

plt.tight_layout()
plt.savefig(f'/home/zhangxinyu/code/fgws-main/draw/compare_wo/woenhance_{name}.pdf', format="pdf", bbox_inches="tight") #

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

