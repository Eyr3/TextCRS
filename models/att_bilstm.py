import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from utils import print_model_state_dict

from .attention import Attention

class AttBiLSTM(nn.Module):
    """
    Implementation of Attention-based bidirectional LSTM proposed in paper [1].

    Parameters
    ----------
    n_classes : int
        Number of classes

    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    rnn_size : int
        Size of Bi-LSTM

    rnn_layers : int
        Number of layers in Bi-LSTM

    dropout : float
        Dropout

    References
    ----------
    1. "`Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification. \
        <https://www.aclweb.org/anthology/P16-2034.pdf>`_" Peng Zhou, et al. ACL 2016.
    """
    def __init__(self, config, logger, pre_trained_embs=None):
        super(AttBiLSTM, self).__init__()
        self.config = config
        self.logger = logger
        self.pre_trained_embs = pre_trained_embs
        
        # embedding layer
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.embed_size)
        if self.pre_trained_embs is not None and self.config.mode == "train":
        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
            self.embedding.weight.data.copy_(torch.from_numpy(self.pre_trained_embs))
            self.logger.log.info("Init emb with pre-trained")
            if self.config.keep_embeddings_fixed:
                self.embedding.weight.requires_grad = False
        
        # self.set_embeddings(self.embeddings, fine_tune)

        # bidirectional LSTM
        self.BiLSTM = nn.LSTM(
            self.config.embed_size, 
            self.config.rnn_size,
            self.config.num_layers,
            bidirectional=True,
            dropout=(0 if self.config.num_layers == 1 else self.config.dropout_rate),
            batch_first=True
        )

        self.attention = Attention(self.config.rnn_size)
        self.fc = nn.Linear(self.config.rnn_size, self.config.num_classes)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.softmax = nn.Softmax(dim=1)

        print_model_state_dict(logger, self)

    def set_embeddings(self, embeddings: torch.Tensor, fine_tune: bool = False) -> None:
        """
        Set weights for embedding layer

        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings

        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            # initialize embedding layer with the uniform distribution
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        text : torch.Tensor (batch_size, word_pad_len)
            Input data

        words_per_sentence : torch.Tensor (batch_size)
            Sentence lengths

        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores
        """
        # word embedding, apply dropout
        # temp = self.embeddings(text)
        # embeds = temp + torch.randn_like(temp, device='cuda') * 1 #self.opt.noise_sd
        embeddings = self.dropout(self.embeddings(text))  # (batch_size, word_pad_len, emb_size)

        # pack sequences (remove word-pads, SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(
            embeddings,
            lengths=words_per_sentence.tolist(),
            batch_first=True,
            enforce_sorted=False
        )  # a PackedSequence object, where 'data' is the flattened words (n_words, emb_size)

        # run through bidirectional LSTM (PyTorch automatically applies it on the PackedSequence)
        rnn_out, _ = self.BiLSTM(packed_words)  # a PackedSequence object, where 'data' is the output of the LSTM (n_words, 2 * rnn_size)

        # unpack sequences (re-pad with 0s, WORDS -> SENTENCES)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)  # (batch_size, word_pad_len, 2 * word_rnn_size)

        # eq.8: h_i = [\overrightarrow{h}_i ⨁ \overleftarrow{h}_i ]
        # H = {h_1, h_2, ..., h_T}
        H = rnn_out[ :, :, : self.config.rnn_size] + rnn_out[ :, :, self.config.rnn_size : ] # (batch_size, word_pad_len, rnn_size)

        # attention module
        r, alphas = self.attention(H)  # (batch_size, rnn_size), (batch_size, word_pad_len)

        # eq.12: h* = tanh(r)
        h = self.tanh(r)  # (batch_size, rnn_size)

        scores = self.fc(self.dropout(h))  # (batch_size, n_classes)

        return scores #, alphas
