import numpy as np
from utils import (
    load_pkl,
    save_pkl,
    bootstrap_sample,
    get_ci,
    get_attack_data,
    load_model,
    compute_adversarial_word_overlap,
)
from config import Config
from logger import Logger
from data_module import DataModule
from detect_utils.detector import Detector
from detect_utils.transformer import Transformer
from models.cnn import CNN
from models.lstm import LSTM
from models.bert_wrapper import BertWrapper


def tune_gamma():
    differences = []

    for orig_text, _ in zip(data_module.val_texts, data_module.val_pols):
        transformed = Transformer(
            orig_text,
            model,
            detector,
            data_module,
            config,
            bert_wrapper=bert_wrapper,
        )
        differences.append(transformed.diff)

    differences.sort()
    p = config.fp_threshold * len(differences)
    thr_idx = int(p) - 1 if p.is_integer() else int(p)
    gamma = differences[thr_idx]

    logger.log.info(
        "num > gamma: {}".format(len([x for x in differences if x > gamma]))
    )
    logger.log.info(
        "num <= gamma: {}".format(len([x for x in differences if x <= gamma]))
    )
    logger.log.info(
        "Fin param tuning: gamma={},thr={}".format(gamma, config.fp_threshold)
    )

    return gamma


def run_detection():
    re_identified_mods_ratio = []
    pr_identified_mods_ratio = []
    all_unperturbed = []
    all_perturbed = []
    neg = 0
    pos = 0
    t_p = 0
    f_p = 0
    t_n = 0
    f_n = 0
    orig_correct = 0
    orig_rob_correct = 0
    adversarial_correct = 0
    adversarial_rob_correct = 0
    valid_advs = 0
    valid_non_advs = 0

    gamma = tune_gamma()

    for idx, (adv_zipped, orig_cl, orig_text) in enumerate(
        zip(adv_examples, attack_pols, attack_sequences)
    ):
        logger.log.info(
            "======= Sample {} of {} =======".format(
                idx + 1, min(len(attack_sequences), config.limit)
            )
        )

        orig_transformed = Transformer(
            orig_text,
            model,
            detector,
            data_module,
            config,
            bert_wrapper=bert_wrapper,
        )

        orig_correct += 1 if orig_cl == orig_transformed.orig_pred else 0
        orig_rob_correct += 1 if orig_cl == orig_transformed.transformed_pred else 0
        valid_non_advs += 1

        orig_transformed.print_info(logger, gamma)
        neg += 1

        if orig_transformed.flipped(gamma):
            f_p += 1
            all_unperturbed.append([orig_transformed.diff, 1])
            logger.log.info("Orig. sample flipped")
        else:
            all_unperturbed.append([orig_transformed.diff, 0])
            t_n += 1
            logger.log.info("Orig. sample not flipped")

        if adv_zipped is not None:
            adv_text = adv_zipped["perturbed"]
            mods = adv_zipped["perturbed_idxs"]

            valid_advs += 1

            adv_transformed = Transformer(
                adv_text,
                model,
                detector,
                data_module,
                config,
                bert_wrapper=bert_wrapper,
            )

            adversarial_correct += 1 if adv_transformed.orig_pred == orig_cl else 0
            adversarial_rob_correct += (
                1 if adv_transformed.transformed_pred == orig_cl else 0
            )

            assert (
                orig_cl == orig_transformed.orig_pred
                and adv_zipped["perturbed_pred"] is not None
            ) or (
                orig_cl != orig_transformed.orig_pred
                and adv_zipped["perturbed_pred"] is None
            )

            if orig_cl == orig_transformed.orig_pred != adv_transformed.orig_pred:
                adv_transformed.print_info(logger, gamma, adversarial=mods)

                pos += 1

                rec, pre = compute_adversarial_word_overlap(
                    mods, adv_transformed.transformed_reps, logger
                )
                re_identified_mods_ratio.append(rec)
                pr_identified_mods_ratio.append(pre)

                if adv_transformed.flipped(gamma):
                    all_perturbed.append([adv_transformed.diff, 1])
                    t_p += 1
                    logger.log.info("Adv. sample flipped")
                else:
                    all_perturbed.append([adv_transformed.diff, 0])
                    f_n += 1
                    logger.log.info("Adv. sample not flipped")

    orig_acc = np.round((orig_correct / valid_non_advs) * 100, 1)
    orig_rob_acc = np.round((orig_rob_correct / valid_non_advs) * 100, 1)
    acc_diff = np.round(((orig_rob_correct - orig_correct) / valid_non_advs) * 100, 1)
    classification_accuracy_adv = (
        np.round((adversarial_correct / valid_advs) * 100, 2) if valid_advs > 0 else 0
    )
    classification_accuracy_rob_adv = (
        np.round((adversarial_rob_correct / valid_advs) * 100, 2)
        if valid_advs > 0
        else 0
    )

    scores_sum = bootstrap_sample(
        all_unperturbed,
        all_perturbed,
        bootstrap_sample_size=config.bootstrap_sample_size,
    )
    c_i_scores = {k: get_ci(v, alpha=config.ci_alpha) for k, v in scores_sum.items()}
    scores = {k: np.mean(v) for k, v in scores_sum.items()}
    scores_rounded = {k: np.round(v * 100, 1) for k, v in scores.items()}

    logger.log.info("============ FINAL RESULTS ============")
    logger.log.info("Non-adversarial sequences: {}".format(valid_non_advs))
    logger.log.info("Adversarial sequences: {}".format(valid_advs))
    logger.log.info("Negatives: {}".format(neg))
    logger.log.info("Positives: {}".format(pos))
    logger.log.info("False Positives: {}".format(f_p))
    logger.log.info("False Negatives: {}".format(f_n))

    for name, score in scores_rounded.items():
        c_i = c_i_scores[name]
        logger.log.info(
            "{}: {} ({}% CI: [{}, {}])".format(
                name.upper(), score, (1 - config.ci_alpha) * 100, c_i[0], c_i[1]
            )
        )

    logger.log.info("Prediction orig ACC: {}".format(orig_acc))
    logger.log.info("Prediction orig rob ACC: {}".format(orig_rob_acc))
    logger.log.info("Orig ACC diff: {}".format(acc_diff))
    logger.log.info(
        "Pr. re-identified perturbed idxs: {}".format(
            np.round(np.mean(pr_identified_mods_ratio) * 100, 1)
        )
    )
    logger.log.info(
        "Re. re-identified perturbed idxs: {}".format(
            np.round(np.mean(re_identified_mods_ratio) * 100, 1)
        )
    )
    logger.log.info(
        "Classification accuracy adv: {}".format(classification_accuracy_adv)
    )
    logger.log.info(
        "Classification accuracy rob adv: {}".format(classification_accuracy_rob_adv)
    )

    return scores["tpr"]


if __name__ == "__main__":
    config = Config()
    model = None
    config.mode = "detect"
    logger = Logger(config)
    data_module = DataModule(config, logger)
    config.vocab_size = len(data_module.vocab)

    attack_sequences, attack_pols = get_attack_data(config, data_module)
    bert_wrapper = None

    if config.use_BERT:
        bert_wrapper = BertWrapper(config, logger)
        model = bert_wrapper.model
    elif config.model_type == "cnn":
        model = CNN(config, logger)
    elif config.model_type == "lstm":
        model = LSTM(config, logger)

    model = load_model(config.load_model_path, model, logger)

    if config.gpu:
        model.cuda()

    adv_examples_path = "{}/attacks/limit_{}/{}/{}/{}/{}/adv_examples.pkl".format(
        config.model_root_path,
        config.limit,
        config.model_type,
        config.attack,
        config.dataset,
        "val_set" if config.detect_val_set else "test_set",
    )

    logger.log.info("Load adversarial examples from {}".format(adv_examples_path))

    adv_examples = load_pkl(adv_examples_path)

    if config.tune_delta_on_val:
        all_vals = []
        best_vals = (0, 0)

        for delta in range(0, 110, 10):
            logger.log.info("Compute TPR for delta={}".format(delta))

            config.delta_thr = delta
            detector = Detector(config, data_module, logger)
            opt_val = run_detection()

            all_vals.append((opt_val, delta))

            if opt_val >= best_vals[0]:
                best_vals = (opt_val, delta)

            logger.log.info("TPR for delta={}: {}".format(delta, opt_val))

        logger.log.info("Optimal TPR at delta={}".format(best_vals[1]))
        logger.log.info("All vals: {}".format(all_vals))

        save_pkl(best_vals[1], config.restore_delta_path)
    else:
        detector = Detector(config, data_module, logger)
        run_detection()
