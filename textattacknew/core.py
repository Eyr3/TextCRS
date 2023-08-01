import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
import sys
sys.path.append("..")
import helper
import textattacknew
import numpy as np
from .certify_K import certify_K


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, tokenizer, training_args, data_module, staircase_mech,
                 nn_matrix=None, word2index=None, index2word=None, mu=None):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.training_args = training_args
        self.data_module = data_module
        self.staircase_mech = staircase_mech
        self.tokenizer = tokenizer

        self.nn_matrix = nn_matrix
        self.word2index = word2index
        self.index2word = index2word
        self.mu = mu
        self.training_args.noise_sd = torch.as_tensor(self.training_args.noise_sd).to(textattacknew.shared.utils.device)

    def scertify(self, x: torch.tensor, n0: int, n: int, alpha: float, ) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """

        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, self.training_args.certify_batch)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, self.training_args.certify_batch)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)

        if self.training_args.num_classes > 2:  # self.training_args.if_addnoise == 4 and
            counts_estimation[cAHat] = 0
            nB = counts_estimation.max()
            pBBar = self._upper_confidence_bound(nB, n, alpha/self.training_args.num_classes)

            if pABar < pBBar:
                return Smooth.ABSTAIN, 0.0, 0.0
            else:
                if self.training_args.if_addnoise in [1, 5]:  # Laplace: -self.lambd * (torch.log(2 * (1 - pABar))
                    radius = max(-0.2*self.training_args.syn_size * np.log(1-pABar+pBBar), 0.1*self.training_args.syn_size*np.log(pABar/pBBar))
                elif self.training_args.if_addnoise in [2, 8]:  # uniform[-lambda, lambda]: 2 * self.lambd * (prob_lb - 0.5)
                    radius = self.training_args.shuffle_len/2 * (pABar - pBBar)
                elif self.training_args.if_addnoise in [3, 7]:
                    radius = self.training_args.sigma / 2 * (norm.ppf(pABar)-norm.ppf(pBBar))
                else:  # if self.training_args.if_addnoise == 4:
                    radius = pABar
                    # radius = certify_K(p_l=pABar, frac_alpha=alpha, global_d=self.training_args.max_len)

                return cAHat, radius, pBBar
        else:  # num_classes = 2
            if pABar < 0.5:
                return Smooth.ABSTAIN, 0.0, 0.0
            else:
                if self.training_args.if_addnoise in [1, 5]:  # Laplace: -self.lambd * (torch.log(2 * (1 - pABar))
                    radius = -0.2 * self.training_args.syn_size * (np.log(2 * (1 - pABar)))
                elif self.training_args.if_addnoise in [2, 8]:  # uniform[-lambda, lambda]: 2 * self.lambd * (prob_lb - 0.5)
                    radius = self.training_args.shuffle_len * (pABar - 0.5)
                elif self.training_args.if_addnoise in [3, 7]:
                    radius = self.training_args.sigma * norm.ppf(pABar)
                else:  # if self.training_args.if_addnoise == 4:
                    radius = pABar
                    # radius = certify_K(p_l=pABar, frac_alpha=alpha, global_d=self.training_args.max_len)

                return cAHat, radius, 0.0

        # noise1-Stairecase: upsilon = 5 / syn_size, lambd = 1 / upsilon = 0.2 * syn_size
        # noise2-Uniform: 2 * lambd = shuffle_len
        # noise3-Gaussian:
        # noise4-Uniform: 2 * lambda = 1 / beta

    def spredict(self, x: torch.tensor, n: int, alpha: float, ) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, self.training_args.certify_batch)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input (one) sentence
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.training_args.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                # input_texts = x.repeat((this_batch_size, 1))  # , 1, 1
                # input_texts = [sen for sen in x for i in range(this_batch_size)]
                input_texts = x * this_batch_size

                input_texts_split = [input_texts[i].split(' ') for i in range(len(input_texts))]
                if self.training_args.if_addnoise == 1:
                    helper.addnoise_1(input_texts_split, self.data_module.dist_mat_neighbor,
                                      self.data_module.word_to_idx,
                                      self.data_module.idx_to_word, self.data_module.dist_mat_idx_to_idx,
                                      self.data_module.idx_to_dist_mat_idx, self.staircase_mech)
                elif self.training_args.if_addnoise == 3:
                    helper.addnoise_3_certify(input_texts_split, self.training_args.pad)
                elif self.training_args.if_addnoise == 4:
                    helper.addnoise_4(input_texts_split, self.training_args.beta, self.training_args.pad)
                elif self.training_args.if_addnoise == 5:
                    helper.addnoise_5_certify(input_texts_split, self.nn_matrix, self.word2index, self.index2word,
                                              self.staircase_mech)
                elif self.training_args.if_addnoise == 6:
                    helper.addnoise_6_certify(input_texts_split, self.nn_matrix, self.word2index, self.index2word,
                                              self.training_args.syn_size)
                elif self.training_args.if_addnoise == 9:  # SAFER
                    helper.addnoise_9(input_texts_split, self.data_module.dist_mat_neighbor,
                                      self.data_module.word_to_idx,
                                      self.data_module.idx_to_word, self.data_module.dist_mat_idx_to_idx,
                                      self.data_module.idx_to_dist_mat_idx, self.training_args.syn_size)
                input_texts = [' '.join(input_texts_split[i]) for i in range(len(input_texts_split))]

                if 'bert' in self.training_args.model_type:
                    # isinstance(self.base_classifier, transformers.PreTrainedModel) or (self.base_classifier, transformers1.PreTrainedModel)
                    input_ids = self.tokenizer(
                        input_texts,
                        padding="max_length",
                        return_tensors="pt",
                        truncation=True,
                    )

                    if self.training_args.if_addnoise in [4, 7, 8]:
                        helper.addnoise_2_bert(input_ids, self.training_args.shuffle_len)

                    input_ids.to(textattacknew.shared.utils.device)

                    if self.training_args.if_addnoise in [3, 7] and 'new' in self.training_args.model_type:
                        logits = self.base_classifier(**input_ids, noise_sd=self.training_args.noise_sd, mu=self.mu)[0]
                        # embeds_init = getattr(self.base_classifier, 'bert').embeddings.word_embeddings(input_ids.data['input_ids'])
                        # embeddings = embeds_init + torch.randn_like(embeds_init).cuda() * self.training_args.sigma
                        # logits = self.base_classifier(input_ids=None, token_type_ids=input_ids.data['token_type_ids'],
                        #                               attention_mask=input_ids.data['attention_mask'], inputs_embeds=embeddings)[0]
                    else:
                        logits = self.base_classifier(**input_ids)[0]
                else:
                    input_ids = self.tokenizer(input_texts)
                    if self.training_args.if_addnoise in [4, 7, 8]:
                        helper.addnoise_2(input_ids, self.training_args.shuffle_len)

                    if not isinstance(input_ids, torch.Tensor):
                        input_ids = torch.tensor(input_ids)

                    input_ids = input_ids.to(textattacknew.shared.utils.device)
                    if self.training_args.if_addnoise in [3, 7] and 'new' in self.training_args.model_type:
                        logits = self.base_classifier(input_ids, noise_sd=self.training_args.noise_sd, mu=self.mu)
                    else:
                        logits = self.base_classifier(input_ids)
                predictions = logits.argmax(dim=-1)
                counts += self._count_arr(predictions.cpu().numpy(), self.training_args.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def _upper_confidence_bound(self, NB: int, N: int, alpha: float):
        """ Returns a (1 - alpha) upper confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NB: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a upper bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NB, N, alpha=2 * alpha, method="beta")[1]
