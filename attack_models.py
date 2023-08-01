import time
import numpy as np
from config import Config
from logger import Logger
from copy import deepcopy
from data_module import DataModule
from models.cnn import CNN
from models.lstm import LSTM
from models.bert_wrapper import BertWrapper
from attacks.genetic_attack import GeneticAttack
from attacks.pwws_attack import PWWSAttack
from attacks.random_attack import RandomAttack
from attacks.prioritized_attack import PrioritizedAttack
from utils import (
    save_pkl,
    inference,
    list_join,
    attack_time_stats,
    load_model,
    AttackArgs,
    get_attack_data,
    copy_file,
    get_oov_count,
)


def attack_model_all_samples(bert_wrapper=None):
    num_samples = len(attack_sequences)
    adversarial_examples = []
    all_num_mods = []
    all_num_mods_ratio = []
    exec_times = []
    total_oov = []
    all_samples = 0
    flipped = 0
    pert_correct = 0
    orig_correct = 0
    attack = None

    if config.attack == "genetic":
        attack = GeneticAttack(
            data_module.language_model, data_module.LM_tokenizer, attack_args
        )
    elif config.attack == "random":
        attack = RandomAttack(attack_args)
    elif config.attack == "pwws":
        attack = PWWSAttack(attack_args)
    elif config.attack == "prioritized":
        attack = PrioritizedAttack(attack_args)
    else:
        logger.log.info("Wrong attack {} specified.".format(config.attack))
        exit()

    for sample in range(num_samples):
        sentence = attack_sequences[sample]
        label = attack_pols[sample]
        len_s = len(sentence)

        if len_s > 0 and (len_s > 1 or config.attack != "genetic"):
            logger.log.info(
                "========= Manipulate sample {} of {} =========".format(
                    sample + 1, num_samples
                )
            )
            logger.log.info("Input sequence length: {}".format(len_s))

            all_samples += 1
            start_t = time.time()

            pred, prob = inference(
                list_join(sentence),
                model,
                data_module.word_to_idx,
                config,
                tokenizer=data_module.spacy_tokenizer,
                bert_wrapper=bert_wrapper,
                single=True,
            )
            prob = prob[pred]

            logger.log.info("Orig label: {}".format(label))
            logger.log.info("Orig prediction: {} ({})".format(pred, np.round(prob, 4)))

            if pred == label:
                orig_correct += 1
                sub_rate = None
                attack_out = attack.attack(
                    sentence, label, model, bert_wrapper=bert_wrapper
                )

                if config.attack == "pwws":
                    (
                        sentence,
                        perturbed_sentence,
                        perturbed_indices,
                        perturbed_prob,
                        perturbed_pred,
                        sub_rate,
                    ) = attack_out
                else:
                    (
                        sentence,
                        perturbed_sentence,
                        perturbed_indices,
                        perturbed_prob,
                        perturbed_pred,
                    ) = attack_out

                perturbed_indices = [
                    (sentence[i], perturbed_sentence[i], i) for i in perturbed_indices
                ]

                adversarial_examples.append(
                    {
                        "clean": sentence,
                        "perturbed": perturbed_sentence,
                        "clean_pred": pred,
                        "label": label,
                        "perturbed_pred": perturbed_pred,
                        "perturbed_idxs": perturbed_indices,
                    }
                )

                num_mods = len(perturbed_indices)
                oov_count = get_oov_count(perturbed_indices, data_module.word_to_idx)
                total_oov.append(oov_count)
                logger.log.info(
                    "OOV ratio for example: {}/{}".format(
                        oov_count, len(perturbed_indices)
                    )
                )

                pert_correct += 1 if perturbed_pred == label else 0
                flipped += 1 if perturbed_pred != label else 0

                highlighted_perturbed = deepcopy(sentence)

                for o, p, i in perturbed_indices:
                    highlighted_perturbed[i] = "[{}]({})".format(p, o)

                logger.log.info("Sentence: {}".format(list_join(sentence)))
                logger.log.info(
                    "Perturbed_sentence: {}".format(list_join(highlighted_perturbed))
                )
                logger.log.info("Label: {}".format(label))
                logger.log.info("Prediction: {} ({})".format(pred, np.round(prob, 4)))
                logger.log.info(
                    "Perturbed prediction: {} ({})".format(
                        perturbed_pred, np.round(perturbed_prob, 4)
                    )
                )
                logger.log.info("Number of modifications to input: {}".format(num_mods))
                logger.log.info("Perturbed indices: {}".format(list(perturbed_indices)))

                all_num_mods_ratio.append(
                    sub_rate if config.attack == "pwws" else num_mods / len_s
                )
                all_num_mods.append(num_mods)

                logger.log.info(
                    "Avg num of modifications ratio: {}".format(
                        np.mean(all_num_mods_ratio)
                    )
                )
            else:
                adversarial_examples.append(
                    {
                        "clean": sentence,
                        "perturbed": sentence,
                        "clean_pred": pred,
                        "label": label,
                        "perturbed_pred": None,
                        "perturbed_idxs": [],
                    }
                )

            if orig_correct > 0:
                logger.log.info("Orig {}/{} correct".format(orig_correct, all_samples))
                logger.log.info("Pert {}/{} correct".format(flipped, orig_correct))
                logger.log.info("ASR: {}".format(np.round(flipped / orig_correct, 4)))
                logger.log.info(
                    "Acc clean: {}".format(np.round(orig_correct / all_samples, 4))
                )
                logger.log.info(
                    "Acc perturbed: {}".format(np.round(pert_correct / all_samples, 4))
                )

            curr_attack_time = time.time() - start_t
            exec_times.append(curr_attack_time)
            attack_time_stats(
                logger, exec_times, curr_attack_time, num_samples - sample
            )
        else:
            adversarial_examples.append(None)

    logger.log.info(
        "================== Attack performance on test set =================="
    )
    logger.log.info("{}/{} correctly classified".format(orig_correct, all_samples))
    logger.log.info(
        "Perturbed {}/{} of correctly classified".format(pert_correct, orig_correct)
    )
    logger.log.info("ASR: {}".format(np.round(flipped / orig_correct, 4)))
    logger.log.info("Acc. clean: {}".format(np.round(orig_correct / all_samples, 4)))
    logger.log.info(
        "Acc. perturbed: {}".format(np.round(pert_correct / all_samples, 4))
    )
    logger.log.info(
        "M (SD) # of modifications: {} ({})".format(
            np.mean(all_num_mods), np.std(all_num_mods)
        )
    )
    logger.log.info(
        "M (SD) oov terms replaced: {} ({})".format(
            np.mean(total_oov), np.std(total_oov)
        )
    )

    save_pkl(adversarial_examples, "{}/adv_examples.pkl".format(config.model_base_path))
    logger.log.info("Saved examples. Done.")


if __name__ == "__main__":
    config = Config()
    logger = Logger(config)
    data_module = DataModule(config, logger)
    config.vocab_size = len(data_module.vocab)

    copy_file(config)

    attack_sequences, attack_pols = get_attack_data(config, data_module)

    model = None
    bert_wrapper = None

    if config.use_BERT:
        bert_wrapper = BertWrapper(config, logger)
        model = bert_wrapper.model
    elif config.model_type == "cnn":
        model = CNN(config, logger)
    elif config.model_type == "lstm":
        model = LSTM(config, logger)
    else:
        logger.log.info("Model Error. Exit.")
        exit()

    model = load_model(config.load_model_path, model, logger)

    if config.gpu:
        model.cuda()

    attack_args = AttackArgs(
        config, logger, data_module.word_to_idx, data_module.spacy_tokenizer
    )

    attack_model_all_samples(bert_wrapper=bert_wrapper)
