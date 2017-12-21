import os
import time

import bhabana.utils as utils

from sacred import Experiment
from bhabana.trainer import Trainer
from bhabana.datasets import IMDB
from bhabana.pipeline import EmbeddingNgramCNNRNNRegression
from torch.optim import adam

ex = Experiment('sentiment_analysis')

@ex.config
def my_config():

    experiment_name = "SA_EMBED_NGRAM_CNN_RNN"
    dataset = {
        "name": "IMDB",
        "n_workers": 4,
        "use_spacy_vocab": False,
        "load_spacy_vectors": True,
        "spacy_model_name": None,
        "cuda": True,
        "rescale": None,
    }
    setup = {
        "name": "brahmaputra",
        "epochs": 20,
        "batch_size": 64,
        "max_time_steps": 30,
        "experiment_name": "{}_{}".format(experiment_name, time.time()),
        "experiment_root_dir": None,
        "evaluate_every": 100,
        "save_every": 100,
        "early_stopping_delta": 0,
        "patience": 5,
        "train_on_gpu": True
    }
    pipeline = {
        "embedding_layer": {
            "embedding_dims": 300,
            "embedding_dropout": 0.5,
            "preload_word_vectors": True,
            "train_embeddings": False
        },
        "ngram_cnn": {
            "cnn_kernel_dims": 100,
            "cnn_kernel_sizes": [1, 3, 5, 7],
            "cnn_layers": 1,
            "cnn_dropout": 0.5
        },
        "rnn": {
            "rnn_hidden_size": 100,
            "rnn_layers": 1,
            "bidirectional": True,
            "rnn_dropout": 0.5,
            "cell_type": "lstm"
        },
        "regression": {
            "activation": "relu"
        }
    }
    optimizer = {
        "name": "adam"
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


def get_dataset_class(dataset_config):

    if dataset_config["name"].lower() == "imdb":
        ds = IMDB(mode="regression",
                  use_spacy_vocab=dataset_config["use_spacy_vocab"],
                  load_spacy_vectors=dataset_config["load_spacy_vectors"],
                  spacy_model_name=dataset_config["spacy_model_name"],
                  aux=["word"], cuda=dataset_config["cuda"],
                  rescale=dataset_config["rescale"])
    else:
        raise NotImplementedError("{} dataset has not been "
                                  "implemented".format(dataset_config["name"]))
    return ds


def get_pipeline(pipeline_config, vocab_size, pretrained_word_vectors):
    pipeline = EmbeddingNgramCNNRNNRegression(vocab_size=vocab_size,
          embedding_dims=pipeline_config["embedding_layer"]["embedding_dims"],
          rnn_hidden_size=pipeline_config["rnn"]["rnn_hidden_size"],
          bidirectional=pipeline_config["rnn"]["bidirectional"],
          rnn_cell=pipeline_config["rnn"]["cell_type"],
          rnn_layers=pipeline_config["rnn"]["rnn_layers"],
          cnn_layers=pipeline_config["ngram_cnn"]["cnn_layers"],
          cnn_kernel_dim=pipeline_config["ngram_cnn"]["cnn_kernel_dims"],
          cnn_kernel_sizes=pipeline_config["ngram_cnn"]["cnn_kernel_sizes"],
          padding_idx=0, pretrained_word_vectors=pretrained_word_vectors,
          trainable_embeddings=pipeline_config["embedding_layer"]["train_embeddings"],
          embedding_dropout=pipeline_config["embedding_layer"]["embedding_dropout"],
          regression_activation=pipeline_config["regression"]["activation"],
          cnn_dropout=pipeline_config["ngram_cnn"]["cnn_dropout"],
          rnn_dropout=pipeline_config["ngram_cnn"]["rnn_dropout"])
    return pipeline

@ex.automain
def run_pipeline(experiment_name, dataset, setup, pipeline, experiment_config):
    ds = get_dataset_class(dataset_config=dataset)
    pipeline = get_pipeline(pipeline, vocab_size=dataset.vocab_sizes["word"],
                            pretrained_word_vectors=dataset.w2v)
    trainer = BrahmaputraTrainer(pipeline=pipeline, dataset=ds,
                                 experiment_config=experiment_config,
                                 n_epochs=setup["epochs"],
                                 batch_size=setup["batch_size"],
                                 n_workers=dataset["n_workers"],
                                 early_stopping_delta=setup[
                                     "early_stopping_delta"],
                                 patience=setup["patience"],
                                 save_every=setup["save_every"],
                                 evaluate_every=setup["evaluate_every"],
                                 train_on_gpu=setup["train_on_gpu"])



class BrahmaputraTrainer(Trainer):

    def __init__(self, pipeline, dataset, experiment_config, n_epochs=10,
                 batch_size=64, n_workers=4, early_stopping_delta=0,
                 patience=5, save_every=100, evaluate_every=100,
                 train_on_gpu=True):
        self.pipeline = pipeline
        self.dataset = dataset
        self.experiment_config = experiment_config
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.early_stopping_delta = early_stopping_delta
        self.patience = patience
        self.save_every = save_every
        self.evaluate_every = evaluate_every
        self.train_on_gpu = train_on_gpu

    def _set_optimizer(self):
        pass

    def run(self):
        self.load()

    def train(self, batch):
        pass

    def evaluate(self, batch):
        pass

    def load(self):


    def save(self, epoch, global_step, best_loss):
        checkpoint_dir = os.path
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    def loss_has_improved(self, batch):
        pass