import argparse

from bhabana.trainer import THE_BOOK_OF_EXPERIMENTS
from bhabana.trainer.brahmaputra import ex as brahmaputra
from bhabana.trainer.yamuna import ex as yamuna
from bhabana.trainer.ganga import ex as ganga
from bhabana.trainer.kanyakumari import ex as kanyakumari
from bhabana.trainer.narmada import ex as narmada
from bhabana.trainer.indus import ex as indus


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name', type=str,
                    help='Name of the experiment that you ant to run. E.g., '
                         'brahmaputra, ganga, etc.')
parser.add_argument('--start_from', type=int, default=1,
                    help='Start')
parser.add_argument('--data_parallel', type=bool,
                    help='data parallel')

args = parser.parse_args()

def get_experiment_by_name(name):
    if name == "brahmaputra":
        return brahmaputra
    elif name == "yamuna":
        return yamuna
    elif name == "ganga":
        return ganga
    elif name == "kanyakumari":
        return kanyakumari
    elif name == "narmada":
        return narmada
    elif name == "indus":
        return indus
    else:
        raise NotImplementedError("This ({}) Experiment template has not been "
                                  "implemented".format(name))


if args.name in THE_BOOK_OF_EXPERIMENTS:
    for i_c, config in enumerate(THE_BOOK_OF_EXPERIMENTS[args.name]):
        if i_c >= args.start_from-1:
            config["experiment_name"] = config["experiment_name"] + " _" + \
                                        args.name + "_" + str(i_c)
            if args.data_parallel:
                config["setup"]["data_parallel"] = True
            experiment = get_experiment_by_name(args.name)
            experiment.run(config_updates=config)
else:
    raise NotImplementedError("This ({}) Experiment template has not been "
                              "implemented".format(args.name))
