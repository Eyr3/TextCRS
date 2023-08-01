"""
Attack Recipe Class
========================

"""

from abc import ABC, abstractmethod

from textattacknew import Attack


class AttackRecipe(Attack, ABC):
    """A recipe for building an NLP adversarial attack from the literature."""

    @staticmethod
    @abstractmethod
    def build(model_wrapper, **kwargs):
        """Creates pre-built :class:`~textattacknew.Attack` that correspond to
        attacks from the literature.

        Args:
            model_wrapper (:class:`~textattack.models.wrappers.ModelWrapper`):
                :class:`~textattacknew.models.wrappers.ModelWrapper` that contains the victim model and tokenizer.
                This is passed to :class:`~textattacknew.goal_functions.GoalFunction` when constructing the attack.
            kwargs:
                Additional keyword arguments.
        Returns:
            :class:`~textattacknew.Attack`
        """
        raise NotImplementedError()
