"""
Prioritized ('Gradient') attack from https://www.aclweb.org/anthology/P19-1103/
"""
import random
from copy import deepcopy
from utils import get_word_net_synonyms, list_join
from attacks.attack import Attack


class PrioritizedAttack(Attack):
    def __init__(self, attack_args):
        Attack.__init__(self, attack_args)

    def attack(self, orig_input, label, model, bert_wrapper=None):
        actual_input = deepcopy(orig_input)
        seq_len = len(actual_input)
        valid_idxs = list(range(seq_len))
        max_num_chg = max(1, int(seq_len * self.config.max_alt_ratio))
        pert_idxs = []
        num_chg = 0

        pred, prob = self.inference(
            list_join(actual_input), model, 1 - label, bert_wrapper=bert_wrapper
        )

        while num_chg < max_num_chg:
            idx = random.choice(valid_idxs)
            valid_idxs.remove(idx)

            cands = get_word_net_synonyms(orig_input[idx])

            if len(cands) > 0:
                synonym_chgs = []

                pred_before, prob_before = self.inference(
                    list_join(actual_input), model, label, bert_wrapper=bert_wrapper
                )

                for cand in cands:
                    proposal = [
                        x if idx != jdx else cand for jdx, x in enumerate(actual_input)
                    ]

                    pred_after, prob_after = self.inference(
                        list_join(proposal), model, label, bert_wrapper=bert_wrapper
                    )

                    synonym_chgs.append((proposal, cand, prob_before - prob_after))

                max_proposal, max_cand, diff = sorted(
                    synonym_chgs, key=lambda t: t[2], reverse=True
                )[0]

                n_pred, n_prob = self.inference(
                    list_join(max_proposal), model, 1 - label, bert_wrapper=bert_wrapper
                )

                actual_input = max_proposal
                prob = n_prob
                pred = n_pred
                pert_idxs.append(idx)
                num_chg += 1

                self.logger.log.info(
                    "Prob after {}/{} (len={}) changes: {}".format(
                        num_chg, max_num_chg, seq_len, prob
                    )
                )

                if pred != label:
                    break

        return orig_input, actual_input, pert_idxs, prob, pred
