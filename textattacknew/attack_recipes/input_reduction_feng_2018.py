"""

Input Reduction
====================
(Pathologies of Neural Models Make Interpretations Difficult)

"""
from textattacknew import Attack
from textattacknew.constraints.pre_transformation import (
    MaxModificationRate,
    RepeatModification,
    StopwordModification,
    InputColumnModification,
)
from textattacknew.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattacknew.goal_functions import InputReduction, UntargetedClassification
from textattacknew.search_methods import GreedyWordSwapWIR, GreedySearch
from textattacknew.transformations import WordDeletion

from .attack_recipe import AttackRecipe


class InputReductionFeng2018(AttackRecipe):
    """Feng, Wallace, Grissom, Iyyer, Rodriguez, Boyd-Graber. (2018).

    Pathologies of Neural Models Make Interpretations Difficult.

    https://arxiv.org/abs/1804.07781
    """

    @staticmethod
    def build(model_wrapper):
        # At each step, we remove the word with the lowest importance value until
        # the model changes its prediction.
        transformation = WordDeletion()

        constraints = [RepeatModification(), StopwordModification()]

        # add constraints
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)

        constraints.append(MaxModificationRate(max_rate=0.05, min_threshold=2))  # max_rate=0.1

        # use_constraint = UniversalSentenceEncoder(
        #     threshold=0.8,  # 0.936338023,
        #     metric="cosine",
        #     compare_against_original=True,
        #     window_size=15,
        #     skip_text_shorter_than_window=True,
        # )
        # constraints.append(use_constraint)

        #
        # Goal is untargeted classification
        #
        # goal_function = InputReduction(model_wrapper, maximizable=True)
        goal_function = UntargetedClassification(model_wrapper)

        # "For each word in an input sentence, we measure its importance by the
        # change in the confidence of the original prediction when we remove
        # that word from the sentence."
        #
        # "Instead of looking at the words with high importance values—what
        # interpretation methods commonly do—we take a complementary approach
        # and study how the model behaves when the supposedly unimportant words are
        # removed."
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)
