"""
LSTM 4 Classification
---------------------------------------------------------------------

"""
import json
import os

import torch
from torch import nn as nn

import textattacknew
from textattacknew.model_args import TEXTATTACK_MODELS
from . import GloveEmbeddingLayer
from textattacknew.models.helpers.utils import load_cached_state_dict
from textattacknew.shared import utils


class LSTMForClassification(nn.Module):
    """A long short-term memory neural network for text classification.

    We use different versions of this network to pretrain models for
    text classification.
    """

    def __init__(
        self,
        hidden_size=150,
        depth=1,
        dropout=0.3,
        num_labels=2,
        max_seq_length=128,
        model_path=None,
        emb_layer_trainable=False,
        glove_size=42,
        # noise_sd=0.0,
        # mu=None,
    ):
        super().__init__()
        self._config = {
            "architectures": "LSTMForClassification",
            "hidden_size": hidden_size,
            "depth": depth,
            "dropout": dropout,
            "num_labels": num_labels,
            "max_seq_length": max_seq_length,
            "model_path": model_path,
            "emb_layer_trainable": emb_layer_trainable,
            "glove_size": glove_size,
            # "noise_sd": noise_sd,
            # "mu": mu,
        }
        # if depth <= 1:
        #     # Fix error where we ask for non-zero dropout with only 1 layer.
        #     # nn.module.RNN won't add dropout for the last recurrent layer,
        #     # so if that's all we have, this will display a warning.
        #     dropout = 0
        self.drop = nn.Dropout(dropout)
        self.emb_layer_trainable = emb_layer_trainable
        self.emb_layer = GloveEmbeddingLayer(emb_layer_trainable=emb_layer_trainable, glove_size=glove_size)
        self.word2id = self.emb_layer.word2id

        self.encoder = nn.LSTM(
            input_size=self.emb_layer.n_d,
            hidden_size=hidden_size // 2,
            num_layers=depth,
            # dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        d_out = hidden_size
        self.out = nn.Linear(d_out, num_labels)
        self.tokenizer = textattacknew.models.tokenizers.GloveTokenizer(
            word_id_map=self.word2id,
            unk_token_id=self.emb_layer.oovid,
            pad_token_id=self.emb_layer.padid,
            max_length=max_seq_length,
        )

        if model_path is not None:
            self.load_from_disk(model_path)
            # self.from_pretrained(model_path)
        self.eval()

    def load_from_disk(self, model_path):
        # TODO: Consider removing this in the future as well as loading via `model_path` in `__init__`.
        import warnings

        warnings.warn(
            "`load_from_disk` method is deprecated. Please save and load using `save_pretrained` and `from_pretrained` methods.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.load_state_dict(load_cached_state_dict(model_path))
        self.eval()

    def save_pretrained(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(
            state_dict,
            os.path.join(output_path, "pytorch_model.bin"),
        )
        with open(os.path.join(output_path, "config.json"), "w") as f:
            # tmp = self._config["mu_std"]
            # if self._config["mu_std"] is not None:
            #     self._config["mu_std"] = np.mean(self._config["mu_std"][:, 0])
            json.dump(self._config, f)
        # self._config["mu_std"] = tmp

    @classmethod
    def from_pretrained(cls, name_or_path):
        """Load trained LSTM model by name or from path.

        Args:
            name_or_path (:obj:`str`): Name of the model (e.g. "lstm-imdb") or model saved via :meth:`save_pretrained`.
        Returns:
            :class:`~textattacknew.models.helpers.LSTMForClassification` model
        """
        if name_or_path in TEXTATTACK_MODELS:
            # path = utils.download_if_needed(TEXTATTACK_MODELS[name_or_path])
            path = utils.download_from_s3(TEXTATTACK_MODELS[name_or_path])
        else:
            path = name_or_path

        config_path = os.path.join(path, "config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Default config
            config = {
                "architectures": "LSTMForClassification",
                "hidden_size": 150,
                "depth": 1,
                "dropout": 0.3,
                "num_labels": 2,
                "max_seq_length": 128,
                "model_path": None,
                "emb_layer_trainable": True,
                "glove_size": 42,
                # "noise_sd": 0.0,
                # "mu": None,
            }
        del config["architectures"]
        model = cls(**config)
        state_dict = load_cached_state_dict(path)
        model.load_state_dict(state_dict)
        return model

    def forward(self, _input, noise_sd=0):
        device = _input.device
        # ensure RNN module weights are part of single contiguous chunk of memory
        self.encoder.flatten_parameters()

        emb = self.emb_layer(_input.t())

        # g_noise = noise_sd * torch.randn_like(emb, device=device)  # \
                  # + torch.mul(mu, torch.ones(emb.size(), device=device)).to(torch.float32)
        # emb = emb + g_noise

        # emb = self.drop(emb)

        output, hidden = self.encoder(emb)
        output = torch.max(output, dim=0)[0]

        output = self.drop(output)
        pred = self.out(output)
        return pred

    def get_input_embeddings(self):
        return self.emb_layer.embedding
