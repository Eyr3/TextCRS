import os
import random
import spacy
from spacy.lang.en import English
import torch
import sklearn
import csv
import numpy as np
from utils import clean_str, cut_raw, save_pkl, load_pkl, shuffle_lists
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import copy


class DataModule:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.logger.log.info("Initialize DataModule")

        random.seed(self.config.seed_val)
        np.random.seed(self.config.seed_val)
        torch.manual_seed(self.config.seed_val)
        spacy.util.fix_random_seed(self.config.seed_val)

        if self.config.gpu:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.train_texts = []
        self.train_raw = []
        self.train_pols = []
        self.val_texts = []
        self.val_raw = []
        self.val_pols = []
        self.test_texts = []
        self.test_raw = []
        self.test_pols = []
        self.vocab = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = {}
        self.init_pretrained = None
        self.language_model = None
        self.LM_tokenizer = None

        nlp = English()
        self.spacy_tokenizer = nlp.tokenizer

        if self.config.mode == "train":
            if self.config.dataset == "imdb":
                self.populate_imdb()
            elif self.config.dataset == "sst2":
                self.populate_sst2()
            elif self.config.dataset == 'amazon':
                self.populate_amazon()
            elif self.config.dataset == 'agnews':
                self.populate_agnews()

            self.set_vocab()

            if config.model_type in ["lstm", "newlstm", "cnn", "newcnn"]:
                self.prepare_init_embs()

            self.logger.log.info("Save data for restore")
            self.save_data()

        # elif self.config.mode == "test":
        #     self.test_amazon()
        #     #self.test_imdb()

        elif self.config.mode not in ["certify", "predict"]:
            self.logger.log.info("Load existing data")
            self.load_data()

        if self.config.mode in ["train", "certify", "predict"]:
            self.print_data_stats()
            if self.config.if_addnoise in [1, 9]:
                self.embeddings = []
                self.dist_mat = []
                self.dist_mat_neighbor = []
                self.idx_to_dist_mat_idx = {}
                self.dist_mat_idx_to_idx = {}
                self.populate_neighbor_embeddings()

        if self.config.mode == "detect":
            self.embeddings = []
            self.dist_mat = []
            self.idx_to_dist_mat_idx = {}
            self.dist_mat_idx_to_idx = {}
            self.populate_embeddings()

        if self.config.mode == "attack" and self.config.attack == "genetic":
            self.load_LM()

    def prepare_init_embs(self):
        """
        Source: https://github.com/nesl/nlp_adversarial_examples (modified)
        """
        if not os.path.exists(self.config.path_to_pre_trained_init):
            self.logger.log.info("Compute pre-trained init embs")
            self.logger.log.info(
                "Loading GloVe Model from {}".format(self.config.glove_path)
            )
            f = open(self.config.glove_path, "r")
            self.init_pretrained = []
            model = {}
            inv = 0

            for line in f:
                row = line.strip().split(" ")
                #print(f"glove example:{row}")
                word = row[0]
                embedding = np.array([float(val) for val in row[1:]])
                #print(f"embedding example:{embedding}")
                model[word] = embedding

            self.logger.log.info("Done. {} words loaded.".format(len(model)))

            for _, word in enumerate(self.vocab):
                try:
                    self.init_pretrained.append(model[word])
                except KeyError:
                    inv += 1
                    self.init_pretrained.append(np.random.uniform(-0.5, 0.5, 300))

            self.init_pretrained = np.array(self.init_pretrained)
            np.save(self.config.path_to_pre_trained_init, self.init_pretrained)

            self.logger.log.info(
                "Finished emb. comp. {}/{} OOV".format(
                    inv, np.shape(self.init_pretrained)[0]
                )
            )
        else:
            self.logger.log.info(
                "Load pre-trained init embs from {}".format(
                    self.config.path_to_pre_trained_init
                )
            )
            self.init_pretrained = np.load(self.config.path_to_pre_trained_init, allow_pickle=True)

    def load_LM(self):
        self.logger.log.info("Load GPT-2 language model")

        self.LM_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.language_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.language_model.eval()

        if self.config.gpu:
            if self.config.use_BERT and int(torch.cuda.device_count()) == 2:
                self.language_model.to(torch.cuda.current_device())  # torch.device("cuda:1")
            else:
                self.language_model.cuda()

    def populate_imdb(self):
        splits = [
            ("train", self.train_texts, self.train_pols, self.train_raw),
            ("test", self.test_texts, self.test_pols, self.test_raw),
        ]

        for (mode, texts, pols, raw_texts) in splits:
            for pol in ["pos", "neg"]:
                path = "{}/{}/{}".format(self.config.path_to_imdb, mode, pol)
                files = os.listdir(path)

                for _, file in enumerate(files):
                    with open("{}/{}".format(path, file), "r") as f:
                        line = f.readlines()[0]
                        raw_texts.append(line)
                        texts.append(
                            cut_raw(
                                clean_str(line, tokenizer=self.spacy_tokenizer),
                                self.config.max_len,
                            )
                        )
                        pols.append(1 if pol == "pos" else 0)

        self.train_texts, self.train_pols, self.train_raw = shuffle_lists(
            self.train_texts, self.train_pols, self.train_raw
        )
        # self.test_texts, self.test_pols, self.test_raw = shuffle_lists(
        #     self.test_texts, self.test_pols, self.test_raw
        # )

        val_thr = int(len(self.train_texts) * (1-self.config.val_split_size))

        self.val_texts = self.train_texts[val_thr:]
        self.val_raw = self.train_raw[val_thr:]
        self.val_pols = self.train_pols[val_thr:]

        self.train_texts = self.train_texts[:val_thr]
        self.train_raw = self.train_raw[:val_thr]
        self.train_pols = self.train_pols[:val_thr]

    def populate_embeddings(self):
        self.logger.log.info("Use pre-trained embeddings")

        if not os.path.exists(self.config.path_to_dist_mat):
            self.logger.log.info("Prepare dist mat")
            self.prepare_embeddings()
        else:
            self.logger.log.info(
                "Load embeddings from {}".format(self.config.path_to_dist_mat)
            )
            self.dist_mat = np.load(self.config.path_to_dist_mat, allow_pickle=True)
            self.idx_to_dist_mat_idx = load_pkl(
                self.config.path_to_idx_to_dist_mat_idx_dict
            )
            self.dist_mat_idx_to_idx = load_pkl(
                self.config.path_to_dist_mat_idx_to_idx_dict
            )

    def populate_neighbor_embeddings(self):
        self.logger.log.info("Use pre-trained neighbor embeddings")

        if not os.path.exists(self.config.path_to_dist_mat_neighbor):
            self.logger.log.info("Prepare neighbor dist mat")
            self.prepare_embeddings()
        else:
            self.logger.log.info(
                "Load top 250 neighbor embeddings from {}".format(self.config.path_to_dist_mat_neighbor)
            )
            self.dist_mat_neighbor = np.load(self.config.path_to_dist_mat_neighbor, allow_pickle=True)
            self.idx_to_dist_mat_idx = load_pkl(
                self.config.path_to_idx_to_dist_mat_idx_dict
            )
            self.dist_mat_idx_to_idx = load_pkl(
                self.config.path_to_dist_mat_idx_to_idx_dict
            )

    def save_data(self):
        base_path = "{}/data".format(self.config.model_base_path)

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        save_pkl(self.train_texts, "{}/{}".format(base_path, "train_texts.pkl"))
        save_pkl(self.val_texts, "{}/{}".format(base_path, "val_texts.pkl"))
        save_pkl(self.test_texts, "{}/{}".format(base_path, "test_texts.pkl"))
        save_pkl(self.train_pols, "{}/{}".format(base_path, "train_pols.pkl"))
        save_pkl(self.val_pols, "{}/{}".format(base_path, "val_pols.pkl"))
        save_pkl(self.test_pols, "{}/{}".format(base_path, "test_pols.pkl"))
        save_pkl(self.train_raw, "{}/{}".format(base_path, "train_raw.pkl"))
        save_pkl(self.val_raw, "{}/{}".format(base_path, "val_raw.pkl"))
        save_pkl(self.test_raw, "{}/{}".format(base_path, "test_raw.pkl"))
        save_pkl(self.vocab, "{}/{}".format(base_path, "vocab.pkl"))
        save_pkl(self.word_to_idx, "{}/{}".format(base_path, "word_to_idx.pkl"))
        save_pkl(self.idx_to_word, "{}/{}".format(base_path, "idx_to_word.pkl"))
        save_pkl(self.word_freq, "{}/{}".format(base_path, "word_freq.pkl"))

    def load_data(self):
        base_path = "{}/data".format(self.config.restore_model_path)
        self.logger.log.info("Load saved data from {}".format(base_path))

        self.train_texts = list(load_pkl("{}/{}".format(base_path, "train_texts.pkl")))
        self.train_pols = list(load_pkl("{}/{}".format(base_path, "train_pols.pkl")))
        self.val_texts = list(load_pkl("{}/{}".format(base_path, "val_texts.pkl")))
        self.val_pols = list(load_pkl("{}/{}".format(base_path, "val_pols.pkl")))
        self.test_texts = list(load_pkl("{}/{}".format(base_path, "test_texts.pkl")))
        self.test_pols = list(load_pkl("{}/{}".format(base_path, "test_pols.pkl")))
        self.train_raw = list(load_pkl("{}/{}".format(base_path, "train_raw.pkl")))
        self.val_raw = list(load_pkl("{}/{}".format(base_path, "val_raw.pkl")))
        self.test_raw = list(load_pkl("{}/{}".format(base_path, "test_raw.pkl")))
        self.vocab = list(load_pkl("{}/{}".format(base_path, "vocab.pkl")))
        self.word_to_idx = load_pkl("{}/{}".format(base_path, "word_to_idx.pkl"))
        self.idx_to_word = load_pkl("{}/{}".format(base_path, "idx_to_word.pkl"))
        self.word_freq = load_pkl("{}/{}".format(base_path, "word_freq.pkl"))

        self.logger.log.info("Vocab size: {}".format(len(self.vocab)))

    def prepare_embeddings(self):
        """
        Source: https://github.com/nesl/nlp_adversarial_examples (modified)
        """
        self.logger.log.info("Loading {} Model".format("glove"))
        f = open(self.config.glove_path, "r")
        model = {}

        for line in f:
            row = line.strip().split(" ")
            word = row[0]
            embedding = np.array([float(val) for val in row[1:]])
            model[word] = embedding

        self.logger.log.info("Done. {} words loaded.".format(len(model)))
        embs = []
        embs_count = 0

        for i, w in enumerate(self.vocab):
            try:
                embs.append(model[w])
                self.idx_to_dist_mat_idx[i] = embs_count
                self.dist_mat_idx_to_idx[embs_count] = i
                embs_count += 1
            except KeyError:
                self.idx_to_dist_mat_idx[i] = False

        self.dist_mat = sklearn.metrics.pairwise.cosine_distances(embs)

        self.logger.log.info("Dist mat shape {}".format(np.shape(self.dist_mat)))
        self.logger.log.info("Save mat")

        save_pkl(self.idx_to_dist_mat_idx, self.config.path_to_idx_to_dist_mat_idx_dict)
        save_pkl(self.dist_mat_idx_to_idx, self.config.path_to_dist_mat_idx_to_idx_dict)
        np.save(self.config.path_to_dist_mat, self.dist_mat, allow_pickle=True)

        self.logger.log.info("Saved mat")

        if self.config.if_addnoise in [1, 9]:
            neighbors_idx = [[] for i in range(len(self.dist_mat))]
            for idx in range(len(self.dist_mat)):
                neighbors = self.dist_mat[idx]
                sorted_idx = np.argsort(neighbors).tolist()
                # sorted_idx.remove(idx)
                neighbors_idx[idx] = sorted_idx[:250]
            np.save(self.config.path_to_dist_mat_neighbor, neighbors_idx, allow_pickle=True)
            self.logger.log.info("Saved mat top 250 neighbors")

    def set_vocab(self):
        word_count = 0

        for line in self.train_texts:
            for word in line:
                try:
                    self.word_freq[word] += 1
                except KeyError:
                    self.word_freq[word] = 1

        freq_words = {}

        for word, freq in self.word_freq.items():
            try:
                freq_words[freq].append(word)
            except KeyError:
                freq_words[freq] = [word]

        sorted_freq_words = sorted(freq_words.items(), reverse=True)
        word_lists = [wl for (_, wl) in sorted_freq_words]
        all_sorted = []

        for wl in word_lists:
            all_sorted += sorted(wl)

        self.vocab.append(self.config.unk_token)
        self.word_to_idx[self.config.unk_token] = word_count
        self.idx_to_word[word_count] = self.config.unk_token
        word_count += 1

        self.vocab.append(self.config.pad_token)
        self.word_to_idx[self.config.pad_token] = word_count
        self.idx_to_word[word_count] = self.config.pad_token
        word_count += 1

        for word in all_sorted:
            self.vocab.append(word)
            self.word_to_idx[word] = word_count
            self.idx_to_word[word_count] = word
            word_count += 1

        self.logger.log.info("Vocab size: {}".format(len(self.vocab)))

    def populate_sst2(self):
        splits = [
            ("train", self.train_texts, self.train_pols, self.train_raw),
            ("dev", self.val_texts, self.val_pols, self.val_raw),
            ("test", self.test_texts, self.test_pols, self.test_raw),
        ]

        for (split, texts, labels, texts_raw) in splits:
        #for (split, labels, texts, texts_raw) in splits:
            with open("{}/{}.tsv".format(self.config.path_to_sst2, split)) as f:
                data = csv.reader(f, delimiter="\t")
                header = False

                for line in data:
                    if not header:
                        header = True
                    else:    
                        if split == 'test':
                            label = line[0][0]
                            text = line[0][2:]
                            texts_raw.append(text)
                            cleaned = cut_raw(
                                clean_str(text, tokenizer=self.spacy_tokenizer),
                                self.config.max_len,
                            )
                            texts.append(cleaned)
                            labels.append(int(label.replace("\n", "").strip()))
                    
                        else:
                            [text, label] = line
                            texts_raw.append(text)
                            cleaned = cut_raw(
                                clean_str(text, tokenizer=self.spacy_tokenizer),
                                self.config.max_len,
                            )
                            texts.append(cleaned)
                            labels.append(int(label.replace("\n", "").strip()))

        self.val_texts, self.val_pols, self.val_raw = shuffle_lists(
            self.val_texts, self.val_pols, self.val_raw
        )
        # self.test_texts, self.test_pols, self.test_raw = shuffle_lists(
        #     self.test_texts, self.test_pols, self.test_raw
        # )

    def populate_amazon(self):
        splits = [
            ("train", self.train_texts, self.train_pols, self.train_raw),
            ("test", self.test_texts, self.test_pols, self.test_raw),
        ]
        
        for (split, texts, labels, texts_raw) in splits:  #for (split, labels, texts, texts_raw) in splits:
            with open("{}/{}.ft.txt".format(self.config.path_to_amazon, split)) as f:
                # data = csv.reader(f, delimiter="\t")
                t = 0

                for line in f:
                    t += 1
                    if t > 50000:
                        break
                       
                    if split == 'train':
                        
                        label = line[9]
                        text = line[11:]
                        
                        texts_raw.append(text)
                        cleaned = cut_raw(
                            clean_str(text, tokenizer=self.spacy_tokenizer),
                            self.config.max_len,
                        )
                        texts.append(cleaned)
                        labels.append((1 if label == "2" else 0))
                
                    else:
                        label = line[9]
                        text = line[11:]
                        #[label, text] = line
                        texts_raw.append(text)
                        cleaned = cut_raw(
                            clean_str(text, tokenizer=self.spacy_tokenizer),
                            self.config.max_len,
                        )
                        texts.append(cleaned)
                        labels.append((1 if label == "2" else 0))
                        
        self.train_texts, self.train_pols, self.train_raw = shuffle_lists(
            self.train_texts, self.train_pols, self.train_raw)
        # self.test_texts, self.test_pols, self.test_raw = shuffle_lists(
        #     self.test_texts, self.test_pols, self.test_raw
        # )

        val_thr = int(len(self.train_texts) * (1-self.config.val_split_size))

        self.val_texts = self.train_texts[val_thr:]
        self.val_raw = self.train_raw[val_thr:]
        self.val_pols = self.train_pols[val_thr:]

        self.train_texts = self.train_texts[:val_thr]
        self.train_raw = self.train_raw[:val_thr]
        self.train_pols = self.train_pols[:val_thr]

    def test_amazon(self):
        splits = [("test", self.test_texts, self.test_pols, self.test_raw),]
        
        for (split, texts, labels, texts_raw) in splits:
            with open("{}/{}.ft.txt".format(self.config.path_to_amazon, split)) as f:
            # with open('/data/ZQData/github-oa/gaus/data/amazon_reviews/test_1.txt') as f:
                # data = csv.reader(f, delimiter="\t")                               
                t = 0
                for line in f:
                    # t += 1
                    # if t > 50000:
                    #     break

                    label = line[9]
                    text = line[11:]
                    texts_raw.append(text)
                    cleaned = cut_raw(
                        clean_str(text, tokenizer=self.spacy_tokenizer),
                        self.config.max_len,
                    )
                    texts.append(cleaned)
                    labels.append((1 if label == "2" else 0))

        # self.test_texts, self.test_pols, self.test_raw = shuffle_lists(
        #     self.test_texts, self.test_pols, self.test_raw
        # )

    def populate_agnews(self):
        splits = [
            ("train", self.train_texts, self.train_pols, self.train_raw),
            ("test", self.test_texts, self.test_pols, self.test_raw),
        ]
        
        for (mode, texts, labels, texts_raw) in splits:
            path = "{}/{}.csv".format(self.config.path_to_agnews, mode)

            with open(path,'r', encoding='utf-8') as f:
                # data = csv.reader(f, delimiter="\t")
                reader = csv.reader(f)
                for line in reader:
                    label = line[0]
                    text = line[2]

                    texts_raw.append(text)
                    cleaned = cut_raw(
                            clean_str(text, tokenizer=self.spacy_tokenizer),
                            self.config.max_len,
                        )
                    texts.append(cleaned)
                    if label == '1':
                        labels.append(0)
                    elif label == '2':
                        labels.append(1)
                    elif label == '3':
                        labels.append(2)
                    elif label == '4':
                        labels.append(3)

        self.train_texts, self.train_pols, self.train_raw = shuffle_lists(
            self.train_texts, self.train_pols, self.train_raw
        )
        # self.test_texts, self.test_pols, self.test_raw = shuffle_lists(
        #     self.test_texts, self.test_pols, self.test_raw
        # )

        val_thr = int(len(self.train_texts) * (1-self.config.val_split_size))

        self.val_texts = self.train_texts[val_thr:]
        self.val_raw = self.train_raw[val_thr:]
        self.val_pols = self.train_pols[val_thr:]

        self.train_texts = self.train_texts[:val_thr]
        self.train_raw = self.train_raw[:val_thr]
        self.train_pols = self.train_pols[:val_thr]

    def test_imdb(self):
        splits = [("test", self.test_texts, self.test_pols, self.test_raw),]

        for (mode, texts, pols, raw_texts) in splits:
            with open('/data/ZQData/github-oa/gaus/data/imdb/test_1.txt') as f:

                for line in f:
                    line = line.split('\t')
                    raw_texts.append(line[1])
                    texts.append(
                        cut_raw(
                            clean_str(line[1], tokenizer=self.spacy_tokenizer),
                            self.config.max_len,
                        )
                    )
                    pols.append(1 if line[0] == "pos" else 0)

        # self.test_texts, self.test_pols, self.test_raw = shuffle_lists(
        #     self.test_texts, self.test_pols, self.test_raw
        # )

    def print_data_stats(self):
        train_dist = ", ".join(
            ["{}: {}".format(cl, self.train_pols.count(cl)) for cl in [0, 1]]
        )
        self.logger.log.info(
            "Train size: {} ({})".format(len(self.train_texts), train_dist)
        )

        val_dist = ", ".join(
            ["{}: {}".format(cl, self.val_pols.count(cl)) for cl in [0, 1]]
        )
        self.logger.log.info("Val size: {} ({})".format(len(self.val_texts), val_dist))

        test_dist = ", ".join(
            ["{}: {}".format(cl, self.test_pols.count(cl)) for cl in [0, 1]]
        )
        self.logger.log.info(
            "Test size: {} ({})".format(len(self.test_texts), test_dist)
        )
