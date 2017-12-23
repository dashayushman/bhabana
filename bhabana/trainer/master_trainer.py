import argparse

from bhabana.trainer import THE_BOOK_OF_EXPERIMENTS
from bhabana.trainer.brahmaputra import ex as brahmaputra


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name', type=str,
                    help='Name of the experiment that you ant to run. E.g., '
                         'brahmaputra, ganga, etc.')
parser.add_argument('--start_from', type=int,
                    help='Start')

args = parser.parse_args()


if args.name in THE_BOOK_OF_EXPERIMENTS:
    for i_c, config in enumerate(THE_BOOK_OF_EXPERIMENTS[args.name]):
        if i_c >= args.start_from-1:
            config["experiment_name"] = config["experiment_name"] + str(i_c)
            brahmaputra.run(config_updates=config)
else:
    raise NotImplementedError("This ({}) Experiment template has not been "
                              "implemented".format(args.name))