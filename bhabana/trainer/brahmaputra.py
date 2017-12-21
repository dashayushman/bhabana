import os
import time

import bhabana.utils as utils
from sacred import Experiment

ex = Experiment('sentiment_analysis')

@ex.config
def my_config():

    model = "brahmaputra"
    dataset = {
        "name": "IMDB",
        "n_workers": 4,
        "use_spacy_vocab": False,
        "load_spacy_vectors": True,
        "mode": "regression",
        "cuda": True,
        "rescale": None,
    }
    setup = {
        "epochs": 20,
        "batch_size": 64,
        "max_time_steps": 30,
        "experiment_name": "{}_{}".format(model, time.time()),
        "experiment_root_dir": None,
        "evaluate_every": 100,
        "save_every": 100,
        "early_stopping_delta": 0.04,
        "patience": 5
    }
    embedding_layer = {
        "vocab_size": None,
        "embedding_dims": 300,
        "embedding_dropout": 0.5,
        "preload_word_vectors": True,
        "train_embeddings": True
    }
    ngram_cnn = {
        "cnn_kernel_dims": 100,
        "cnn_kernel_sizes": [1, 3, 5, 7],
        "cnn_layers": 1,
        "cnn_dropout": 0.5
    }
    rnn = {
        "rnn_hidden_size": 100,
        "rnn_layers": 1,
        "bidirectional": True,
        "rnn_dropout": 0.5,
        "cell_types": "lstm"
    }
    regression = {
        "activation": None
    }
    experiment_config = experiment_boilerplate(setup)

def experiment_boilerplate(setup_config):
    experiment_config = {}
    if setup_config["experiment_root_dir"] is None:
        experiment_config["experiment_root_dir"] = utils.EXPERIMENTS_DIR
        setup_config["experiment_root_dir"] = utils.EXPERIMENTS_DIR
    experiment_config["experiment_dir"] = os.path.join(
            experiment_config["experiment_root_dir"], setup_config["experiment_name"])
    experiment_config["checkpoints_dir"] = os.path.join(
            experiment_config["experiment_dir"], "checkpoints")
    experiment_config["logs_dir"] = os.path.join(
            experiment_config["experiment_dir"], "logs")
    experiment_config["test_results"] = os.path.join(
            experiment_config["experiment_dir"], "test_results.tsv")
    experiment_config["val_results"] = os.path.join(
            experiment_config["experiment_dir"], "val_results.tsv")
    experiment_config["train_results"] = os.path.join(
            experiment_config["experiment_dir"], "train_results.tsv")


    if not os.path.exists(experiment_config["experiment_dir"]):
        os.makedirs(experiment_config["experiment_dir"])

    if not os.path.exists(experiment_config["checkpoints_dir"]):
        os.makedirs(experiment_config["checkpoints_dir"])
    if not os.path.exists(experiment_config["logs_dir"]):
        os.makedirs(experiment_config["logs_dir"])
    return experiment_config



@ex.automain
def run_pipeline(dataset, pipeline, setup, embedding_layer, ngram_cnn, rnn,
                 regression):
    pass


class BrahmaputraTrainer(Trainer):

def train():
    pass


def evaluate():
    pass


def save_model():
    pass
