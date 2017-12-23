import argparse

from bhabana.trainer import THE_BOOK_OF_EXPERIMENTS
from bhabana.trainer.brahmaputra import ex as brahmaputra


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name', type=str,
                    help='Name of the experiment that you ant to run. E.g., '
                         'brahmaputra, ganga, etc.')

args = parser.parse_args()


if args.n in THE_BOOK_OF_EXPERIMENTS:
    for config in THE_BOOK_OF_EXPERIMENTS[args.name]:
        brahmaputra.run(config_updates=config)
else:
    raise NotImplementedError("This ({}) Experiment template has not been "
                              "implemented".format(args.name))