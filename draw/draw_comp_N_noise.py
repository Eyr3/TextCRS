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
font = {'serif' : 'Times New Roman',
        # 'weight' : 'bold',
        'size'   : 23}
matplotlib.rc('font', **font)
print(matplotlib.rcParams)

Accs = \
{'lstm': \
[83.8, 84.0, 84.6, 84.8, # textfooler  #83, 83.2, 83.8, 84
91.2, 91.2, 91.2, 91.2, # wordreorder
60.8, 62.4, 63, 63.2, # synonyn insert
75,75.2, 76.4, 76.6, # input_reduction
],
'bert': \
[87.4, 87.6, 88, 88,
84.6, 84.6, 84.6, 84.6,
72.2, 72.8, 73, 73,
77.8, 78.2, 78.6, 78.6,
],
}

fig, ax = plt.subplots(nrows=1, ncols=2, layout='constrained')  # constrained_layout=True
fig.set_size_inches(15, 4.4)

X = ['Text\nFooler','Word\nReorder','Synonym\nInsert', 'Input\nReduction'] 
X_axis = np.arange(len(X))  # the label locations
bar_width = 0.15
offset = bar_width/2
opacity=1

ax1 = plt.subplot(1, 2, 1)
data = np.array(Accs['lstm']).reshape((4,4))/100
ax1.bar(X_axis-bar_width-offset, data[:,0], bar_width, color='#4c72b0', alpha=opacity, label='N=10,000')
ax1.bar(X_axis-offset, data[:,1], bar_width, color='#dd8452', alpha=opacity, label='N=20,000')
ax1.bar(X_axis+offset, data[:,2], bar_width, color='#55a868', alpha=opacity, label='N=100,000')
ax1.bar(X_axis+bar_width+offset, data[:,3], bar_width, color='#c44e52', alpha=opacity, label='N=200,000')
ax1.set_title('LSTM', fontsize=23)
ax1.set_xticks(np.arange(len(X)), X, rotation=0)
# ax1.set_xlabel("Dataset")
ax1.set_ylim(0.55, 0.95)
ax1.set_ylabel('Certified accuracy') # Certified accuracy  Cert. Acc.


ax2 = plt.subplot(1, 2, 2)
data = np.array(Accs['bert']).reshape((4,4))/100
ax2.bar(X_axis-bar_width-offset, data[:,0], bar_width, color='#4c72b0', alpha=opacity, label='N=10,000')
ax2.bar(X_axis-offset, data[:,1], bar_width, color='#dd8452', alpha=opacity, label='N=20,000')
ax2.bar(X_axis+offset, data[:,2], bar_width, color='#55a868', alpha=opacity, label='N=100,000')
ax2.bar(X_axis+bar_width+offset, data[:,3], bar_width, color='#c44e52', alpha=opacity, label='N=200,000')
ax2.set_title('BERT', fontsize=23)
ax2.set_xticks(np.arange(len(X)), X, rotation=0)
# ax2.set_xlabel("Dataset")
ax2.set_ylim(0.55, 0.95)
ax2.set_ylabel('Certified accuracy')

# ax.legend(loc='bottom right', ncol=2)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor =(0.5,-0.09), loc='lower center', ncol=4)
# fig.legend(lines, labels, loc = (0.5, 0), ncol=5)


plt.show()

plt.tight_layout()
plt.savefig(f'/home/zhangxinyu/code/fgws-main/draw/compare_N/compare_noise.png', bbox_inches="tight") #

exit()




name = 'lstm_benign'
if 'lstm' in name:
    data = np.array(Accs['lstm']).reshape((9,4))
else:
    data = np.array(Accs['bert']).reshape((9,4))
ylim_range={'lstm_benign':[60, 96],  'lstm_certify':[50, 90],
            'bert_benign':[80, 98],  'bert_certify':[60, 90],
            }

X = ['$\sigma$=0.1','$\sigma$=0.2','$\sigma$=0.3', '$\sigma$=0.1','$\sigma$=0.2','$\sigma$=0.3', '$\sigma$=0.1','$\sigma$=0.2','$\sigma$=0.3'] 
bar_width = 0.25
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
    plt.bar(X_axis - offset, acc_wo, bar_width, color='#1f77b4', alpha=0.8, label='Benign Acc._w/o enhance')
    plt.bar(X_axis + offset, acc_w, bar_width, color='#e19153', alpha=opacity, label='Benign Acc._w/ enhance')

else:
    # plt.bar(X_axis - offset, cer_acc_wo, bar_width, hatch='//', edgecolor='#2ca02c', color='white', alpha=opacity, label='Cert. Acc._w/o enhance')
    # plt.bar(X_axis + offset, cer_acc_w, bar_width, hatch='+', edgecolor='#d62728', color='white', alpha=opacity, label='Cert. Acc._w/ enhance')
    plt.bar(X_axis - offset, cer_acc_wo, bar_width, color='#48b2a3', alpha=opacity, label='Cert. Acc._w/o enhance')
    plt.bar(X_axis + offset, cer_acc_w, bar_width, color='#da6046', alpha=opacity, label='Cert. Acc._w/ enhance')

plt.axvline(2.5, linestyle='-', linewidth=5, c='white')
plt.axvline(5.5, linestyle='-', linewidth=5, c='white')

# plt.axvline(2.5, linestyle=':', linewidth=3, c='gray')
# plt.axvline(5.5, linestyle=':', linewidth=3, c='gray')

ax.set_ylim(ylim_range[name][0], ylim_range[name][1])
ax.set_xticks(np.arange(len(X)), X)
ax.set_xlabel("Smoothing Paramter (Gaussian Noise Level)")
if 'benign' in name:
    ax.set_ylabel("Benign Acc. (%)")
else:    
    ax.set_ylabel("Cert. Acc. (%)")
ax.set_title("AG                          Amazon                          IMDB")
if 'lstm' in name:
    ax.legend(loc='upper right', ncol=2)
else:
    ax.legend(loc='upper left', ncol=2)
plt.show()

plt.tight_layout()
plt.savefig(f'/home/zhangxinyu/code/fgws-main/draw/compare_wo/woenhance_{name}.png', bbox_inches="tight") #

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
