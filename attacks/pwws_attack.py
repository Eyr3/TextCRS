"""
Implementation of Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency
Paper: https://www.aclweb.org/anthology/P19-1103/
Parts adapted from https://github.com/JHL-HUST/PWWS/
"""
import spacy
from utils import list_join
from attacks.attack import Attack
from attacks.pwws_paraphrase import PWWS, compile_perturbed_tokens


class PWWSAttack(Attack):
    def __init__(self, attack_args):
        Attack.__init__(self, attack_args)
        self.nlp = spacy.load("en_core_web_sm")

    def attack(self, orig_input, label, model, bert_wrapper=None):
        tag = self.nlp(list_join(orig_input))

        orig_pred, orig_prob = self.inference(
            tag.text, model, label, bert_wrapper=bert_wrapper
        )
        word_saliencies = self.word_saliencies(
            tag, model, label, orig_prob, bert_wrapper=bert_wrapper
        )

        perturbed, sub_rate, NE_rate, change_tuple_list = list(
            PWWS(
                tag,
                label,
                self.config.dataset,
                self.config.max_len,
                model,
                self.logger,
                bert_wrapper=bert_wrapper,
                word_saliency_list=word_saliencies,
                heuristic_fn=self.pwws_heuristic_fn,
                halt_condition_fn=self.pwws_halt_condition_fn,
            )
        )

        perturbed = [t.text for t in perturbed]
        pred, prob = self.inference(
            list_join(perturbed), model, 1 - label, bert_wrapper=bert_wrapper
        )
        perturbed_indices = [x[0] for x in change_tuple_list]

        return [t.text for t in tag], perturbed, perturbed_indices, prob, pred, sub_rate

    def word_saliencies(self, tag, model, label, orig_prob, bert_wrapper=None):
        word_saliencies = []

        for idx, word in enumerate(tag):
            masked = [self.unk_token if idx == i else x.text for i, x in enumerate(tag)]
            _, masked_prob = self.inference(
                list_join(masked), model, label, bert_wrapper=bert_wrapper
            )
            word_saliencies.append((idx, word, orig_prob - masked_prob, word.tag_))

        return word_saliencies

    def pwws_heuristic_fn(self, orig, candidate, label, model, bert_wrapper=None):
        perturbed = self.nlp(list_join(compile_perturbed_tokens(orig, [candidate])))
        _, prob_orig = self.inference(
            orig.text, model, label, bert_wrapper=bert_wrapper
        )
        _, prob_pert = self.inference(
            perturbed.text, model, label, bert_wrapper=bert_wrapper
        )
        return prob_orig - prob_pert

    def pwws_halt_condition_fn(self, actual_input, model, label, bert_wrapper=None):
        pred, prob = self.inference(
            actual_input.text, model, 1 - label, bert_wrapper=bert_wrapper
        )
        return pred != label, prob
