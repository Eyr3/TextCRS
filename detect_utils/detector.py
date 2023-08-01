import math
import random
import numpy as np
from utils import load_pkl, get_word_net_synonyms, get_freq, get_neighboring_embeddings


class Detector:
    def __init__(self, config, data_module, logger):
        self.config = config
        self.data_module = data_module
        self.logger = logger
        self.word_to_idx = self.data_module.word_to_idx
        self.f_all = data_module.word_freq
        self.baseline = self.config.detect_baseline
        self.f_all_log = {}
        self.freq_threshold = 0

        self.load_freqs()

        self.n_neighbors = self.get_n_neighbors()
        self.logger.log.info("Value for K: {}".format(self.n_neighbors))

        if not self.baseline:
            self.get_delta_threshold()

    def get_delta_threshold(self):
        if self.config.delta_thr is None:
            try:
                self.config.delta_thr = load_pkl(self.config.restore_delta_path)
            except FileNotFoundError:
                self.logger.log.info("Delta not optimized. Exit.")
                exit()

        self.freq_threshold = np.percentile(
            sorted(self.f_all_log), self.config.delta_thr
        )
        self.logger.log.info(
            "Delta thr: {}, freq threshold: {}".format(
                self.config.delta_thr, self.freq_threshold
            )
        )
        percentile = len([x for x in self.f_all_log if x < self.freq_threshold]) / len(
            self.f_all_log
        )
        self.logger.log.info("Percentile: {}".format(percentile))

    def get_n_neighbors(self):
        words = {}

        for text in self.data_module.val_texts:
            for word in text:
                if self.valid_word(word):
                    words[word] = get_word_net_synonyms(word)

        return int(math.ceil(np.mean([len(v) for _, v in words.items()])))

    def load_freqs(self):
        self.f_all_log = [
            get_freq(self.f_all, k, use_log=True)
            for k in self.f_all.keys()
            if self.valid_word(k)
        ]

    def valid_word(self, word):
        return word != self.config.eos_token

    def detector_module(self, input_seq):
        if self.baseline:
            low_freq_idxs = [
                idx
                for idx, w in enumerate(input_seq)
                if self.valid_word(w) and get_freq(self.f_all, w) == 0
            ]
        else:
            input_freqs = [
                get_freq(self.f_all, word, use_log=True) for word in input_seq
            ]
            valid_freqs_idxs = [
                True if x < self.freq_threshold else False for x in input_freqs
            ]
            valid_freqs_idxs = [
                x and self.valid_word(w) for x, w in zip(valid_freqs_idxs, input_seq)
            ]
            low_freq_idxs = [idx for idx, x in enumerate(valid_freqs_idxs) if x is True]

        replaced_idx = []

        for idx in low_freq_idxs:
            word = input_seq[idx]

            neighbors = get_word_net_synonyms(word)

            try:
                if (
                    self.data_module.idx_to_dist_mat_idx[self.word_to_idx[word]]
                    is not False
                ):
                    embed_neighbors = get_neighboring_embeddings(
                        word,
                        self.data_module.dist_mat,
                        self.word_to_idx,
                        self.data_module.idx_to_word,
                        self.n_neighbors,
                        self.data_module.dist_mat_idx_to_idx,
                        self.data_module.idx_to_dist_mat_idx,
                    )

                    neighbors = list(set(neighbors + embed_neighbors))
            except KeyError:
                pass

            if self.baseline:
                neighbors = [
                    w for w in neighbors if get_freq(self.f_all, w, use_log=False) > 0
                ]

            if len(neighbors) > 0:
                if self.baseline:
                    rep = random.choice(neighbors)
                    input_seq[idx] = rep
                    replaced_idx.append((word, rep, idx))
                else:
                    neighbors = {
                        x: get_freq(self.f_all, x, use_log=True) for x in neighbors
                    }

                    # See https://stackoverflow.com/a/15371752
                    neighbors = [
                        w
                        for (w, _) in sorted(
                            neighbors.items(), key=lambda x: (x[1], x[0]), reverse=True
                        )
                    ]

                    rep = neighbors[0]

                    if get_freq(self.f_all, rep, use_log=True) > get_freq(
                        self.f_all, word, use_log=True
                    ):
                        input_seq[idx] = rep
                        replaced_idx.append((word, rep, idx))

        return input_seq, replaced_idx
