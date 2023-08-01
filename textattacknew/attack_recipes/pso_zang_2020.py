"""

Particle Swarm Optimization
==================================

(Word-level Textual Adversarial Attacking as Combinatorial Optimization)

"""
from textattacknew import Attack
from textattacknew.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattacknew.goal_functions import UntargetedClassification
from textattacknew.search_methods import ParticleSwarmOptimization
from textattacknew.transformations import WordSwapHowNet

from .attack_recipe import AttackRecipe


class PSOZang2020(AttackRecipe):
    """Zang, Y., Yang, C., Qi, F., Liu, Z., Zhang, M., Liu, Q., & Sun, M.
    (2019).

    Word-level Textual Adversarial Attacking as Combinatorial Optimization.

    https://www.aclweb.org/anthology/2020.acl-main.540.pdf

    Methodology description quoted from the paper:

    "We propose a novel word substitution-based textual attack model, which reforms
    both the aforementioned two steps. In the first step, we adopt a sememe-based word
    substitution strategy, which can generate more candidate adversarial examples with
    better semantic preservation. In the second step, we utilize particle swarm optimization
    (Eberhart and Kennedy, 1995) as the adversarial example searching algorithm."

    And "Following the settings in Alzantot et al. (2018), we set the max iteration time G to 20."
    """

    @staticmethod
    def build(model_wrapper):
        #
        # Swap words with their synonyms extracted based on the HowNet.
        #
        transformation = WordSwapHowNet()
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        #
        # Use untargeted classification for demo, can be switched to targeted one
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Perform word substitution with a Particle Swarm Optimization (PSO) algorithm.
        #
        search_method = ParticleSwarmOptimization(pop_size=60, max_iters=20)

        return Attack(goal_function, constraints, transformation, search_method)
