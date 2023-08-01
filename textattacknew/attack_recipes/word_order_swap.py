"""
Randomly Swaps the Order of Words in A Sequence
============================================

"""

from textattacknew import Attack
from textattacknew.constraints.pre_transformation import (
    MaxModificationRate,
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattacknew.transformations import WordInnerSwapRandom
from textattacknew.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattacknew.goal_functions import UntargetedClassification
from textattacknew.search_methods import GreedySearch

from .attack_recipe import AttackRecipe


class WordOrderSwap(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        transformation = WordInnerSwapRandom()

        constraints = [RepeatModification(), StopwordModification()]  # stopwords=stopwords

        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        constraints.append(MaxModificationRate(max_rate=0.1, min_threshold=4))
        # constraints.append(PartOfSpeech(allow_verb_noun_swap=False))

        use_constraint = UniversalSentenceEncoder(
            threshold=0.8,  # 0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)

        goal_function = UntargetedClassification(model_wrapper)

        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
