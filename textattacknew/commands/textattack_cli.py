"""

TextAttack CLI main class
==============================

"""


# !/usr/bin/env python
import argparse

from textattacknew.commands.attack_command import AttackCommand
from textattacknew.commands.attack_resume_command import AttackResumeCommand
from textattacknew.commands.augment_command import AugmentCommand
from textattacknew.commands.benchmark_recipe_command import BenchmarkRecipeCommand
from textattacknew.commands.eval_model_command import EvalModelCommand
from textattacknew.commands.list_things_command import ListThingsCommand
from textattacknew.commands.peek_dataset_command import PeekDatasetCommand
from textattacknew.commands.train_model_command import TrainModelCommand


def main():
    parser = argparse.ArgumentParser(
        "TextAttack CLI",
        usage="[python -m] texattack <command> [<args>]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(help="textattacknew command helpers")

    # Register commands
    AttackCommand.register_subcommand(subparsers)
    AttackResumeCommand.register_subcommand(subparsers)
    AugmentCommand.register_subcommand(subparsers)
    BenchmarkRecipeCommand.register_subcommand(subparsers)
    EvalModelCommand.register_subcommand(subparsers)
    ListThingsCommand.register_subcommand(subparsers)
    TrainModelCommand.register_subcommand(subparsers)
    PeekDatasetCommand.register_subcommand(subparsers)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    func = args.func
    del args.func
    func.run(args)


if __name__ == "__main__":
    main()
