"""
Parts adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py
"""
import torch
import torch.nn as nn
from utils import print_model_state_dict


class LSTM(nn.Module):
    def __init__(self, config, logger, pre_trained_embs=None):
        super(LSTM, self).__init__()
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

        self.lstm = nn.LSTM(
            self.config.embed_size,
            self.config.hidden_size//2,
            self.config.num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = nn.Linear(self.config.hidden_size, self.config.num_classes)

        print_model_state_dict(logger, self)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        if self.config.if_addnoise == 3:
            embeddings = embeddings + self.config.noise_sd * torch.randn_like(embeddings).cuda()  #, device=torch.cuda.current_device()
        out, _ = self.lstm(embeddings)
        # out = torch.mean(out, 1)
        out = torch.max(out, dim=1)[0].squeeze()
        out = self.dropout(out)
        out = self.fc(out)

        return out
