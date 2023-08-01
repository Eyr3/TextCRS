"""
Parts adapted from https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import print_model_state_dict


class CNN(nn.Module):
    def __init__(self, config, logger, pre_trained_embs=None):
        super(CNN, self).__init__()
        self.config = config
        self.logger = logger
        self.pre_trained_embs = pre_trained_embs

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embed_size)

        if self.pre_trained_embs is not None and self.config.mode == "train":
            # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
            self.embedding.weight.data.copy_(torch.from_numpy(self.pre_trained_embs))
            self.logger.log.info("Init emb with pre-trained")

            if self.config.keep_embeddings_fixed:
                self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    self.config.num_feature_maps,
                    (fs, self.config.embed_size),
                    self.config.stride,
                )
                for fs in self.config.filter_sizes
            ]
        )

        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = nn.Linear(
            len(self.config.filter_sizes) * self.config.num_feature_maps,
            self.config.num_classes,
        )

        print_model_state_dict(logger, self)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        out = embeddings.unsqueeze(1)
        out = [F.relu(conv(out)).squeeze(3) for conv in self.convs]
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1)
        out = self.dropout(out)
        out = self.fc(out)

        return out
