import os
import json
import torch
import shutil
import codecs
import collections

import numpy as np
import torch.nn as nn
import bhabana.utils as utils

from torch import optim
from sacred import Experiment
from bhabana.datasets import IMDB
from bhabana.trainer import Trainer
from torch.autograd import Variable
from bhabana.processing import Id2Seq
from tensorboardX import SummaryWriter
from sacred.observers import SlackObserver
from sacred.observers import MongoObserver
from bhabana.metrics import Accuracy
from bhabana.metrics import FMeasure
from bhabana.metrics import ClassificationReport
from torch.optim.lr_scheduler import MultiStepLR
from bhabana.pipeline import EmbeddingNgramCNNRNNClassification



slack_config_file_path = os.path.join(os.path.dirname(__file__), "slack.json")
mongo_config_file_path = os.path.join(os.path.dirname(__file__), "mongo.json")

with open(mongo_config_file_path, "r") as jf:
    mongo_config = json.load(jf)
    mongo_obs = MongoObserver.create(
    url='mongodb://{}:{}@{}/sacred_experiments?authMechanism=SCRAM'
        '-SHA-1'.format(mongo_config["user"], mongo_config["password"],
                        mongo_config["url"]),
    db_name='sacred_experiments')

slack_obs = SlackObserver.from_config(slack_config_file_path)
ex = Experiment('sentiment_analysis')
ex.observers.append(slack_obs)
ex.observers.append(mongo_obs)

@ex.config
def my_config():

    experiment_name = "SA_EMBED_NGRAM_CNN_RNN"
    experiment_description = "default experiment"
    dataset = {
        "name": "IMDB",
        "n_workers": 1,
        "use_spacy_vocab": False,
        "load_spacy_vectors": False,
        "spacy_model_name": None,
        "cuda": True,
        "rescale": None,
        "max_seq_length": 100
    }
    setup = {
        "name": "yamuna",
        "epochs": 20,
        "batch_size": 64,
        "experiment_name": "{}".format(experiment_name),
        "experiment_root_dir": None,
        "evaluate_every": 100,
        "save_every": 100,
        "early_stopping_delta": 0,
        "patience": 10,
        "train_on_gpu": True,
        "data_parallel": False
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
            "rnn_hidden_size": 50,
            "rnn_layers": 1,
            "bidirectional": True,
            "rnn_dropout": 0.5,
            "cell_type": "lstm"
        }
    }
    optimizer = {
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "lr_scheduling_milestones": [3, 7, 17]
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
            experiment_config["experiment_dir"], "test_results")
    experiment_config["validation_results"] = os.path.join(
            experiment_config["experiment_dir"], "validation_results")
    experiment_config["train_results"] = os.path.join(
            experiment_config["experiment_dir"], "train_results")


    if not os.path.exists(experiment_config["experiment_dir"]):
        os.makedirs(experiment_config["experiment_dir"])
    if not os.path.exists(experiment_config["checkpoints_dir"]):
        os.makedirs(experiment_config["checkpoints_dir"])
    if not os.path.exists(experiment_config["logs_dir"]):
        os.makedirs(experiment_config["logs_dir"])
    if not os.path.exists(experiment_config["test_results"]):
        os.makedirs(experiment_config["test_results"])
    if not os.path.exists(experiment_config["validation_results"]):
        os.makedirs(experiment_config["validation_results"])
    if not os.path.exists(experiment_config["train_results"]):
        os.makedirs(experiment_config["train_results"])
    return experiment_config


class YamunaTrainer(Trainer):

    def __init__(self, experiment_name, pipeline, dataset, experiment_config,
                 logger, run, n_epochs=10, batch_size=64, max_seq_length=0,
                 n_workers=4, early_stopping_delta=0, patience=5,
                 save_every=100, evaluate_every=100, learning_rate=0.001,
                 weight_decay=0.0, train_on_gpu=True, data_parallel=False,
                 lr_scheduling_milestones=[1, 3, 5, 7]):
        self.experiment_name = experiment_name
        self.logger = logger
        self.sacred_run = run
        self.pipeline = pipeline
        self.data_parallel = data_parallel
        self.pipeline = torch.nn.DataParallel(self.pipeline) if \
                                        self.data_parallel else self.pipeline
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.experiment_config = experiment_config
        self.writer = SummaryWriter(log_dir=experiment_config["logs_dir"])
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.n_workers = n_workers
        self.early_stopping_delta = early_stopping_delta
        self.patience = patience
        self.save_every = save_every
        self.evaluate_every = evaluate_every
        self.train_on_gpu = train_on_gpu and torch.cuda.is_available()
        self.lr_scheduling_milestones = lr_scheduling_milestones
        self._set_optimizer()
        if self.train_on_gpu:
            self.logger.info("CUDA found. Training model on GPU")
            self.pipeline.cuda()
        self.loss = nn.CrossEntropyLoss()
        self.metrics = [Accuracy(), FMeasure(), ClassificationReport()]
        self.best_model_path = os.path.join(self.experiment_config["checkpoints_dir"],
                                            "best_model.pth.tar")
        self._set_trackers()
        self._set_dataloaders()
        self.sequence_decoder = Id2Seq(i2w=self.dataset.vocab["word"][1],
                                       mode="word", batch=True)

    def _set_trackers(self):
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = 100000
        self.no_improvement_since = 0
        self.time_to_stop = False
        self.loss_history = []

    def _set_dataloaders(self):
        self.dataloader = {"train": self.dataset.get_batch(split="train",
                                               to_tensor=True,
                                               pad=True,
                                               shuffle=False,
                                               batch_size=self.batch_size,
                                               num_workers=self.n_workers,
                                               seq_max_len=self.max_seq_length),
                           "validation": self.dataset.get_batch(
                                               split="validation",
                                               to_tensor=True,
                                               pad=True, shuffle=False,
                                               batch_size=self.batch_size,
                                               num_workers=self.n_workers,
                                               seq_max_len=self.max_seq_length),
                           "test": self.dataset.get_batch(split="test",
                                              to_tensor=True,
                                              pad=True,
                                              shuffle=False,
                                              batch_size=self.batch_size,
                                              num_workers=self.n_workers,
                                              seq_max_len=self.max_seq_length)
                           }

    def _set_optimizer(self):
        self.logger.info("Initializing the Optimizer")
        trainable_parameters= filter(lambda p: p.requires_grad,
                                     self.pipeline.parameters())
        self.optimizer = optim.Adam(trainable_parameters,
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        self.scheduler = MultiStepLR(self.optimizer,
                                     milestones=self.lr_scheduling_milestones,
                                     gamma=0.1)

    def __call__(self, *args, **kwargs):
        self.run()

    def run(self):
        self.load()
        for epoch in range(self.current_epoch, self.n_epochs):
            self.pipeline.train()
            self.logger.info("Training Epoch: {}".format(epoch))
            self.current_epoch = epoch
            for i_train, train_batch in self.dataloader["train"]:
                self.train(train_batch)
                self.pipeline.eval()
                if self.time_to_evaluate(self.evaluate_every, i_train):
                    self.logger.info("Evaluating: mode=Validation")
                    self.run_evaluation_epoch(self.dataloader["validation"],
                                              mode="validation")
                if self.time_to_save(self.save_every, i_train):
                    self.save()
                self.pipeline.train()
            self._post_epoch_routine()
            self.scheduler.step(epoch)
            self.current_epoch += 1
            if self.time_to_stop:
                self.closure()
                break

    def _post_epoch_routine(self):
        self.pipeline.eval()
        self.logger.info("Evaluating: mode=Validation")
        self.run_evaluation_epoch(self.dataloader["validation"],
                                  mode="validation", write_results=True)
        self.logger.info("Evaluating: mode=Test")
        self.run_evaluation_epoch(self.dataloader["test"], mode="test",
                                  write_results=True)
        self.save()
        self.log_tf_embeddings()
        self.pipeline.train()
        self.restart_dataloader("train")

    def get_rnn_hidden(self):
        hidden = self.pipeline.init_rnn_hidden(self.batch_size,
                                               self.train_on_gpu)
        return hidden

    def train(self, batch):
        self.pipeline.zero_grad()
        batch["inputs"] = batch["text"]
        batch["training"] = True
        batch["hidden"] = self.get_rnn_hidden()
        output = self.pipeline(batch)
        loss = self.loss(output["out"], torch.max(batch["label"], dim=1)[1])
        scalar_loss = loss.data.cpu().numpy()[0] if self.train_on_gpu else loss.data.numpy()[0]
        loss.backward()
        self.optimizer.step()
        self.log("training.loss", loss)
        self.global_step += 1
        if self.global_step % 10 == 0:
            self.logger.info("Epoch: {}\tGlobal Step: {}\t"
                             "Training Loss: {}".format(self.current_epoch,
                                                self.global_step,
                                                scalar_loss))
            # self.log_histogram()
            pred = output["out"].data.cpu().numpy() if self.train_on_gpu else \
                                                    output["out"].data.numpy()
            gt = batch["label"].data.cpu().numpy() if self.train_on_gpu \
                else batch["sentiment"].data.numpy()
            acc, f_score, clf_rpt = self._run_metrics(pred, gt)
            self.log("training.Accuracy", acc)
            self.log("training.F1_Score", f_score)

    def run_evaluation_epoch(self, dataloader, mode="validation",
                             write_results=False):
        val_losses, val_accs, val_fscores = [], [], []
        for i_val, val_batch in dataloader:
            pred, gt, val_loss, acc, f_score, clf_rpt = self.evaluate(val_batch)
            val_losses.append(val_loss)
            val_accs.append(acc)
            val_fscores.append(f_score)
            if write_results:
                self.write_results_to_file(i_val, val_batch["text"], pred,
                                           gt, mode)
            if i_val % 10 == 0:
                self.logger.info("Epoch: {}\tGlobal Step: {}\t"
                                 "Mode: {}\tLoss: {}\tAcc: {}\t"
                                 "F1-Score: {}\tBatch Number: {}".format(
                        self.current_epoch, self.global_step, mode,
                        val_loss, acc, f_score, i_val))
        avg_loss, avg_acc, avg_f1score = np.average(val_losses),\
                                         np.average(val_accs), \
                                         np.average(val_fscores)
        self.log("{}.Average_Loss".format(mode), avg_loss)
        self.log("{}.Accuracy".format(mode), avg_acc)
        self.log("{}.F1_Score".format(mode), avg_f1score)
        if mode == "validation":
            self.update_loss_history(avg_loss)
        self.restart_dataloader(mode)

    def restart_dataloader(self, mode):
        self.dataloader[mode] = self.dataset.get_batch(split=mode,
              to_tensor=True, pad=True, shuffle=False,
              batch_size=self.batch_size, num_workers=self.n_workers,
              seq_max_len=self.max_seq_length)

    def evaluate(self, batch):
        batch["inputs"] = batch["text"]
        batch["training"] = False
        batch["hidden"] = self.get_rnn_hidden()
        output = self.pipeline(batch)
        loss = self.loss(output["out"], torch.max(batch["label"], dim=1)[1])
        pred = output["out"].data.cpu().numpy() if self.train_on_gpu else \
                                                output["out"].data.numpy()
        gt = batch["label"].data.cpu().numpy() if self.train_on_gpu else \
                                            batch["sentiment"].data.numpy()
        acc, f_score, clf_rpt = self._run_metrics(pred, gt)

        scalar_loss = loss.data.cpu().numpy()[0] if self.train_on_gpu else loss.data.numpy()[0]
        return pred, gt, scalar_loss, acc, f_score, clf_rpt

    def _run_metrics(self, pred, gt):
        results = []
        for metric in self.metrics:
            results.append(metric(pred, gt))
        return results[0], results[1], results[2]

    def load(self):
        self.logger.info("Trying to load checkpoint from '{}'".format(
                self.best_model_path))
        if os.path.exists(self.best_model_path):
            self.logger.info("Found checkpoint. Attemptimg to load it")
            checkpoint = torch.load(self.best_model_path)
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_loss = checkpoint['best_loss']
            self.pipeline.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("Loaded checkpoint from: '{}'\n"
             " Current Epoch: {}\n Best Loss: {}".format(self.best_model_path,
                                 checkpoint['epoch'], checkpoint['best_loss']))
        else:
            self.logger.info(("Could not find checkpoint at : {} "
                              "Training fresh parameters".format(
                    self.best_model_path)))

    def save(self):
        if self.loss_has_improved():
            self.logger.info("Saving Model as the loss has improved")
            checkpoint_path = os.path.join(self.experiment_config[
              "checkpoints_dir"], "model_{}_{}.pth.tar".format(self.current_epoch,
                                                               self.global_step))
            torch.save({
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'arch': self.experiment_name,
                'state_dict': self.pipeline.state_dict(),
                'best_loss': self.best_loss,
                'optimizer': self.optimizer.state_dict(),
            }, checkpoint_path)
            shutil.copyfile(checkpoint_path, self.best_model_path)

    def loss_has_improved(self):
        if self.no_improvement_since == 0:
            return True
        else:
            return False

    def update_loss_history(self, loss):
        if not self.time_to_stop:
            if len(self.loss_history) > 0:
                delta = self.loss_history[-1] - loss
                if delta > self.early_stopping_delta:
                    self.no_improvement_since = 0
                else:
                    self.no_improvement_since += 1
                    if self.no_improvement_since >= self.patience:
                        self.time_to_stop = True
            else:
                self.no_improvement_since = 0

            self.loss_history.append(loss)

    def closure(self):
        self.writer.close()

    def _get_scalar(self, value):
        if type(value) == np.ndarray:
            sacred_value = value
            tf_value = value
        elif type(value) == Variable:
            sacred_value = value.data.cpu().numpy()[0] if \
                self.train_on_gpu else value.data.numpy()[0]
            tf_value = value.data.cpu().numpy() if \
                self.train_on_gpu else value.data.numpy()
        elif type(value) in [float, int, np.float32, np.int, np.int32, np.float64]:
            sacred_value = value
            tf_value = np.expand_dims(np.array(value), axis=1)
        else:
            self.logger.warning("Unable to log values because of unknown "
                                "dtype of the value")
            sacred_value, tf_value = None, None
        return float(sacred_value), tf_value

    def log(self, name, value):
        sacred_value, tf_value = self._get_scalar(value)
        if sacred_value is None or tf_value is None:
            self.logger.warning("Value could not be converted into the right "
                                "format. This needs to be looked at.")
        self.sacred_run.log_scalar(name, sacred_value, self.global_step)
        self.writer.add_scalar(name.replace(".", "/"),  tf_value, self.global_step)

    def log_histogram(self):
        for name, param in self.pipeline.named_parameters():
            param_values = param.clone().cpu().data.numpy() if \
                self.train_on_gpu else param.clone().data.numpy()
            self.writer.add_histogram(name, param_values, self.global_step)

    def write_results_to_file(self, batch_id, batch, pred, gt, mode,
                              tf_text=True):

        file_path = os.path.join(self.experiment_config["{}_results".format(
                mode)], "{}_epoch_{}_batch_{}.tsv".format(mode,
                                      self.current_epoch, batch_id))
        batch = batch.data.cpu().numpy().tolist() if self.train_on_gpu else \
                                            batch.data.numpy().tolist()
        text = self.sequence_decoder(batch)
        with codecs.open(file_path, "w", "utf-8") as rf:
            text_buff = []
            for t, pred, gt in zip(text, pred, gt):
                text_buff.append("{}\t{}\t{}".format(t, pred, gt))
                rf.write(text_buff[-1] + "\n")
                if tf_text:
                    self.writer.add_text(mode, "\n".join(text_buff),
                                         self.global_step)

    def log_tf_embeddings(self):
        self.logger.info("Saving Embeddings for Projector Visualization")
        ordered_i2w = collections.OrderedDict(sorted(self.dataset.vocab["word"][1].items()))
        labels = [val for key, val in ordered_i2w.items()]
        self.writer.add_embedding(self.pipeline.get_embedding_weights(),
                                  labels, global_step=self.global_step,
                                  tag="word_embeddings")


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


def get_pipeline(pipeline_config, vocab_size, pretrained_word_vectors,
                 n_classes):
    pipeline = EmbeddingNgramCNNRNNClassification(vocab_size=vocab_size,
          embedding_dims=pipeline_config["embedding_layer"]["embedding_dims"],
          rnn_hidden_size=pipeline_config["rnn"]["rnn_hidden_size"],
          n_classes=n_classes,
          bidirectional=pipeline_config["rnn"]["bidirectional"],
          rnn_cell=pipeline_config["rnn"]["cell_type"],
          rnn_layers=pipeline_config["rnn"]["rnn_layers"],
          cnn_layers=pipeline_config["ngram_cnn"]["cnn_layers"],
          cnn_kernel_dim=pipeline_config["ngram_cnn"]["cnn_kernel_dims"],
          cnn_kernel_sizes=pipeline_config["ngram_cnn"]["cnn_kernel_sizes"],
          padding_idx=0, pretrained_word_vectors=pretrained_word_vectors,
          trainable_embeddings=pipeline_config["embedding_layer"]["train_embeddings"],
          embedding_dropout=pipeline_config["embedding_layer"]["embedding_dropout"],
          cnn_dropout=pipeline_config["ngram_cnn"]["cnn_dropout"],
          rnn_dropout=pipeline_config["rnn"]["rnn_dropout"])
    return pipeline

@ex.automain
def run_pipeline(experiment_name, dataset, setup, pipeline,
                 optimizer, experiment_config, _log, _run):
    ds = get_dataset_class(dataset_config=dataset)
    pipeline = get_pipeline(pipeline, vocab_size=ds.vocab_sizes["word"],
                            pretrained_word_vectors=ds.w2v, n_classes=ds.n_classes)
    trainer = YamunaTrainer(experiment_name=experiment_name,
                                 pipeline=pipeline, dataset=ds,
                                 experiment_config=experiment_config,
                                 logger=_log,
                                 run=_run,
                                 n_epochs=setup["epochs"],
                                 batch_size=setup["batch_size"],
                                 max_seq_length=dataset["max_seq_length"],
                                 n_workers=dataset["n_workers"],
                                 early_stopping_delta=setup[
                                     "early_stopping_delta"],
                                 patience=setup["patience"],
                                 save_every=setup["save_every"],
                                 evaluate_every=setup["evaluate_every"],
                                 learning_rate=optimizer["learning_rate"],
                                 weight_decay=optimizer["weight_decay"],
                                 train_on_gpu=setup["train_on_gpu"],
                                 data_parallel=setup["data_parallel"],
                                 lr_scheduling_milestones=optimizer["lr_scheduling_milestones"])
    trainer.run()