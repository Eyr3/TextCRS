import torch.nn as nn


class ChannelCompress(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256, dropout=0.5, channel_rate=4):
        """
        reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
        out_ch could be 512, 256, 128
        """
        super(ChannelCompress, self).__init__()
        num_bottleneck = int(in_ch * channel_rate)  # int(in_ch / 2), in_ch
        # half_bottleneck = in_ch  # int(num_bottleneck / 2)
        add_block = []
        add_block += [nn.Linear(in_ch, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.ReLU()]
        add_block += [nn.Dropout(p=dropout)]

        # add_block += [nn.Linear(num_bottleneck, half_bottleneck)]
        # add_block += [nn.BatchNorm1d(half_bottleneck)]
        # add_block += [nn.ReLU()]
        # add_block += [nn.Dropout(p=dropout)]
        add_block += [nn.Linear(num_bottleneck, out_ch)]

        # Extra BN layer, need to be removed
        #add_block += [nn.BatchNorm1d(out_ch)]

        add_block = nn.Sequential(*add_block)
        # add_block.apply(weights_init_kaiming)
        self.channel = add_block

    def forward(self, x):
        x = self.channel(x)
        return x
