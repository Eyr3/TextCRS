"""
Reimplementation of Generating Natural Language Adversarial Examples
Paper: https://arxiv.org/abs/1804.07998
Author source code: https://github.com/nesl/nlp_adversarial_examples
"""
import numpy as np
from numpy.random import choice
from copy import deepcopy
from nltk.corpus import stopwords
from utils import (
    crossover,
    attack_get_neighboring_embeddings,
    get_n_neighbors_delta,
    compute_perplexity_GPT2,
    load_attack_mat,
    list_join,
)
from attacks.attack import Attack


class GeneticAttack(Attack):
    def __init__(self, LM_model, LM_tokenizer, attack_args):
        Attack.__init__(self, attack_args)
        self.LM_model = LM_model
        self.LM_tokenizer = LM_tokenizer
        self.k = self.config.k
        self.LM_cut = self.config.LM_cut
        self.LM_ws = self.config.LM_window_size
        self.num_gen = self.config.num_gen
        self.num_pop = self.config.num_pop
        self.stopwords = stopwords.words("english")
        self.n_neighbors_delta_map = {}
        (
            self.attack_dist_mat,
            self.attack_dist_mat_word_to_idx,
            self.attack_dist_mat_idx_to_word,
        ) = load_attack_mat(self.config, self.logger)

    def attack(self, orig_input, label, model, bert_wrapper=None):
        actual_input = deepcopy(orig_input)
        max_num_chg = max(1, int(len(actual_input) * self.config.max_alt_ratio))
        generations = {i: [] for i in range(self.num_gen + 1)}
        max_adv = None

        # Populate the initial generation
        packed_selections = self.perturb_batch(
            [actual_input] * self.num_pop,
            label,
            model,
            orig_input,
            bert_wrapper=bert_wrapper,
        )

        if packed_selections is None:
            pred, prob = self.inference(
                list_join(actual_input), model, 1 - label, bert_wrapper=bert_wrapper
            )
            return orig_input, actual_input, [], prob, pred
        else:
            for i in range(self.num_pop):
                cand, cand_prob, cand_pred, pert_idx = packed_selections[i]
                generations[0].append(
                    Candidate(
                        cand,
                        cand_prob,
                        cand_pred,
                        [pert_idx] if pert_idx is not None else [],
                    )
                )

        # Populate the following generations
        for g in range(1, self.num_gen + 1):
            sorted_cands = sorted(
                generations[g - 1], key=lambda t: t.confidence, reverse=True
            )
            max_adv = sorted_cands[0]

            self.logger.log.info(
                "Populate generation {}/{} (num pop {}; max prob gen {} was {})".format(
                    g,
                    self.num_gen,
                    self.num_pop,
                    g - 1,
                    np.round(max_adv.confidence, 4),
                )
            )

            assert len(max_adv.perturbed_idx) == len(set(max_adv.perturbed_idx))

            if (
                max_adv.prediction != label
                and len(max_adv.perturbed_idx) <= max_num_chg
            ):
                return (
                    orig_input,
                    max_adv.text,
                    max_adv.perturbed_idx,
                    max_adv.confidence,
                    max_adv.prediction,
                )
            elif len(max_adv.perturbed_idx) >= max_num_chg:
                return (
                    orig_input,
                    max_adv.text,
                    max_adv.perturbed_idx,
                    max_adv.confidence,
                    max_adv.prediction,
                )
            else:
                generations[g].append(max_adv)
                children, children_idx = self.get_children(generations[g - 1])
                packed_selections = self.perturb_batch(
                    children, label, model, orig_input, bert_wrapper=bert_wrapper
                )

                if packed_selections is None:
                    return (
                        orig_input,
                        max_adv.text,
                        max_adv.perturbed_idx,
                        max_adv.confidence,
                        max_adv.prediction,
                    )
                else:
                    for z in range(self.num_pop - 1):
                        cand, cand_prob, cand_pred, pert_idx = packed_selections[z]

                        if pert_idx is not None:
                            children_idx[z].append(pert_idx)

                        generations[g].append(
                            Candidate(
                                cand, cand_prob, cand_pred, list(set(children_idx[z]))
                            )
                        )

        return (
            orig_input,
            max_adv.text,
            max_adv.perturbed_idx,
            max_adv.confidence,
            max_adv.prediction,
        )

    def get_children(self, generation):
        children = []
        children_idx = []

        for _ in range(1, self.num_pop):
            probs_prev = [cand.confidence for cand in generation]

            if sum(probs_prev) != 0:
                normalised_probs = [
                    cand.confidence / sum(probs_prev) for cand in generation
                ]
            else:
                normalised_probs = [1.0 / len(probs_prev)] * len(probs_prev)

            multinomial = np.random.multinomial(2, normalised_probs)

            if len(np.where(multinomial == 1)[0].tolist()) == 0:
                [sample_idx_1, sample_idx_2] = (
                    np.where(multinomial == 2)[0].tolist() * 2
                )
            else:
                [sample_idx_1, sample_idx_2] = np.where(multinomial == 1)[0].tolist()

            child, idx = crossover(generation[sample_idx_1], generation[sample_idx_2])
            children.append(child)
            children_idx.append(idx)

        return children, children_idx

    def perturb_batch(self, inputs, label, model, orig_input, bert_wrapper=None):
        reps = []
        perplexity_reps = []
        idxs = []
        LM_reps = []
        neighbor_lengths = []
        all_neighbors = []

        for actual_input in inputs:
            valids = [
                (idx, x)
                for idx, x in enumerate(actual_input)
                if self.valid_word_to_replace(x)
            ]

            if len(valids) > 0:
                scores = [
                    get_n_neighbors_delta(
                        x,
                        self.attack_dist_mat_word_to_idx,
                        self.attack_dist_mat,
                        self.config.delta,
                        self.n_neighbors_delta_map,
                    )
                    for (_, x) in valids
                ]
                scores = [x / sum(scores) for x in scores]
                idx = choice([idx for (idx, _) in valids], 1, p=scores)[0]
                word = actual_input[idx]

                neighbors = attack_get_neighboring_embeddings(
                    word,
                    self.attack_dist_mat,
                    self.attack_dist_mat_word_to_idx,
                    self.attack_dist_mat_idx_to_word,
                    self.k,
                    orig_input[idx],
                    delta=self.config.delta,
                )

                neighbor_lengths.append(len(neighbors))

                for candidate in neighbors:
                    rep = [
                        actual_input[jdx] if jdx != idx else candidate
                        for jdx in range(len(actual_input))
                    ]
                    perplexity_reps.append(rep)
                    reps.append(rep)
                    idxs.append(idx)
                    suffix = (
                        list_join(actual_input[idx + 1 : idx + 1 + self.LM_ws])
                        if idx < len(actual_input) - 1
                        else ""
                    )
                    prefix = (
                        list_join(actual_input[max(idx - self.LM_ws, 0) : idx])
                        if idx > 0
                        else ""
                    )
                    LM_reps.append((prefix, candidate, suffix))
                    all_neighbors.append(candidate)
            else:
                self.logger.log.info("No valids")
                neighbor_lengths.append(0)

        if len(reps) > 0:
            preds, new_class_probs = self.inference_batch(
                [list_join(x) for x in reps],
                model,
                1 - label,
                bert_wrapper=bert_wrapper,
            )
        else:
            return None

        selections = []
        scores = []

        for prefix, word, suffix in LM_reps:
            scores.append(
                compute_perplexity_GPT2(
                    [list_join([prefix, word, suffix])],
                    self.LM_tokenizer,
                    self.LM_model,
                    is_gpu=self.config.gpu,
                )
            )

        done = 0

        for idx, batch_len in enumerate(neighbor_lengths):
            if batch_len == 0:
                pred, new_class_prob = self.inference(
                    list_join(inputs[idx]), model, 1 - label, bert_wrapper=bert_wrapper
                )
                selections.append((inputs[idx], new_class_prob, pred, None))
            else:
                assert batch_len == neighbor_lengths[idx]

                candidates = list(
                    zip(
                        new_class_probs[done : done + batch_len],
                        reps[done : done + batch_len],
                        preds[done : done + batch_len],
                        idxs[done : done + batch_len],
                        all_neighbors[done : done + batch_len],
                        scores[done : done + batch_len],
                    )
                )

                candidates = sorted(candidates, key=lambda t: t[5], reverse=False)[
                    : self.LM_cut
                ]
                sorted_candidates = sorted(candidates, key=lambda t: t[0], reverse=True)
                (ncp, output, pred, p_idx, b_n, _) = sorted_candidates[0]
                selections.append((output, ncp, pred, p_idx))

                done += batch_len

        return selections

    def valid_word_to_replace(self, word):
        if word not in self.stopwords:
            try:
                _ = self.attack_dist_mat_word_to_idx[word]
                return True
            except KeyError:
                return False

        return False


class Candidate:
    def __init__(self, text, confidence, prediction, perturbed_idx):
        self.text = text
        self.confidence = confidence
        self.prediction = prediction
        self.perturbed_idx = perturbed_idx
