import os
import json
import torch
import shutil
import codecs
import collections

import bhabana.trainer.training_configs as configs
import numpy as np
import torch.nn as nn
import bhabana.utils as utils
import bhabana.utils.constants as constants

from tqdm import tqdm
from torch import optim
from sacred import Experiment
from bhabana.datasets import IMDB
from bhabana.datasets import AmazonReviews
from bhabana.datasets import SentimentTreebank
from bhabana.datasets import KaggleSentiment
from bhabana.trainer import Trainer
from torch.autograd import Variable
from bhabana.processing import Id2Seq
from tensorboardX import SummaryWriter
from sacred.observers import SlackObserver
from sacred.observers import MongoObserver
from bhabana.metrics import PearsonCorrelation
from bhabana.metrics import Accuracy
from bhabana.metrics import FMeasure
from bhabana.metrics import ClassificationReport
from torch.optim.lr_scheduler import MultiStepLR
from bhabana.pipeline import EmbeddingNgramCNNRNNMultiTask
from bhabana.utils import data_utils as du


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
    mode = "train"
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
        "name": "krishna",
        "epochs": 20,
        "batch_size": 64,
        "experiment_name": "{}".format(experiment_name),
        "experiment_root_dir": None,
        "evaluate_every": 100,
        "save_every": 100,
        "early_stopping_delta": 0,
        "patience": 10,
        "train_on_gpu": True,
        "data_parallel": False,
        "eval_test": True,
        "eval_val": True,
        "load_path": None,
        "fine_tuning": False,
        "mode": "train"
    }
    pipeline = {
        "embedding_layer": {
            "embedding_dims": 300,
            "embedding_dropout": 0.1,
            "preload_word_vectors": True,
            "train_embeddings": False
        },
        "ngram_cnn": {
            "cnn_kernel_dims": 100,
            "cnn_kernel_sizes": [1, 3, 5, 7],
            "cnn_layers": 1,
            "cnn_dropout": 0.2
        },
        "rnn": {
            "rnn_hidden_size": 50,
            "rnn_layers": 1,
            "bidirectional": True,
            "rnn_dropout": 0.3,
            "cell_type": "lstm"
        },
        "regression": {
            "activation": "relu"
        }
    }
    optimizer = {
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "lr_scheduling_milestones": [1, 3, 5, 7, 9, 13, 17]
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


class KrishnaTrainer(Trainer):

    def __init__(self, experiment_name, pipeline, dataset, experiment_config,
                 logger, run, n_epochs=10, batch_size=64, max_seq_length=0,
                 n_workers=4, early_stopping_delta=0, patience=5,
                 save_every=100, evaluate_every=100, learning_rate=0.001,
                 weight_decay=0.0, train_on_gpu=True, eval_test=True,
                 eval_val=True, load_path=None, fine_tuning=False):
        self.experiment_name = experiment_name
        self.eval_val = eval_val
        self.eval_test = eval_test
        self.logger = logger
        self.sacred_run = run
        self.pipeline = pipeline
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
        self._set_optimizer()
        if self.train_on_gpu:
            self.logger.info("CUDA found. Training model on GPU")
            self.pipeline.cuda()
        self.loss = nn.MSELoss()
        self.clf_loss = nn.CrossEntropyLoss()
        self.metric = PearsonCorrelation()
        self.clf_metrics = [Accuracy(), FMeasure(), ClassificationReport()]
        self.best_model_path = os.path.join(self.experiment_config["checkpoints_dir"],
                                            "best_model.pth.tar")
        self._set_trackers()
        self.dataloader = {}
        #self._set_dataloaders()
        self.sequence_decoder = Id2Seq(i2w=self.dataset.vocab["word"][1],
                                       mode="word", batch=True)
        self.load_path = load_path
        self.fine_tuning = fine_tuning

    def preprocess(self, text):
        processed_fields = []
        if hasattr(self.dataset.fields[0]["processors"], "__iter__"):
            for processor in self.dataset.fields[0]["processors"]:
                if processor.add_to_output:
                    processed_fields.append(processor(text))
        else:
            if self.dataset.fields[0]["processors"].add_to_output:
                processed_fields.append(self.dataset.fields[0]["processors"](
                        text))
        return processed_fields[0]["text"]

    def predict(self, text, batch_size):
        processed_text = text
        #processed_text.append(constants.PAD)
        if self.train_on_gpu:
            text_tensor = Variable(torch.LongTensor(processed_text).pin_memory().cuda())
        else:
            text_tensor = Variable(torch.LongTensor(processed_text))
        self.pipeline.eval()
        batch = {"inputs": text_tensor, "training": False, "hidden":
            self.get_rnn_hidden(batch_size)}
        output = self.pipeline(batch)
        output["clf_out"] = torch.nn.functional.softmax(output["clf_out"])
        reg_pred = output["reg_out"].data.cpu().numpy() if \
            self.train_on_gpu else output["reg_out"].data.numpy()
        clf_pred = output["clf_out"].data.cpu().numpy() if \
            self.train_on_gpu else output["clf_out"].data.numpy()

        return reg_pred, clf_pred

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

    def restart_dataloader(self, mode):
        self.dataloader[mode] = self.dataset.get_batch(split=mode,
                  to_tensor=True, pad=True, shuffle=False,
                  batch_size=self.batch_size, num_workers=self.n_workers,
                  seq_max_len=self.max_seq_length)

    def _set_optimizer(self):
        self.logger.info("Initializing the Optimizer")
        trainable_parameters= filter(lambda p: p.requires_grad,
                                     self.pipeline.parameters())
        self.optimizer = optim.Adam(trainable_parameters,
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        self.scheduler = MultiStepLR(self.optimizer,
                                     milestones=[60, 100, 150, 400],
                                     gamma=0.1)

    def __call__(self, *args, **kwargs):
        self.run()

    def run(self):
        if self.load_path is not None:
            self.load(self.load_path)
        _once_train , _once_val, _once_test = True, True, True
        for epoch in range(self.current_epoch, self.n_epochs):
            self.pipeline.train()
            self.logger.info("Training Epoch: {}".format(epoch))
            self.current_epoch = epoch
            if _once_train:
                self.restart_dataloader("train")
                _once_train = False
            for i_train, train_batch in self.dataloader["train"]:
                self.train(train_batch)
                self.pipeline.eval()
                if self.time_to_evaluate(self.evaluate_every, i_train) and \
                        self.eval_val:
                    self.logger.info("Evaluating: mode=Validation")
                    if _once_val:
                        self.restart_dataloader("validation")
                        _once_val = False
                    self.run_evaluation_epoch(self.dataloader["validation"],
                                              mode="validation",
                                              write_results=True)
                if self.time_to_save(self.save_every, i_train):
                    self.save()
                self.pipeline.train()
            self._post_epoch_routine(_once_test)
            _once_test = False
            self.scheduler.step(epoch)
            self.current_epoch += 1
            if self.time_to_stop:
                self.closure()
                break

    def test(self):
        self.load()
        self.pipeline.eval()
        self.logger.info("Evaluating: mode=Validation")
        self.restart_dataloader("validation")
        self.run_evaluation_epoch(self.dataloader["validation"],
                                  mode="validation", write_results=True,
                                  write_vectors=True)
        self.logger.info("Evaluating: mode=Test")
        self.restart_dataloader("test")
        self.run_evaluation_epoch(self.dataloader["test"], mode="test",
                                  write_results=True,
                                  write_vectors=True)
        self.save()
        self.pipeline.train()
        self.restart_dataloader("train")

    def _post_epoch_routine(self, once_test):
        self.pipeline.eval()
        if self.eval_val:
            self.logger.info("Evaluating: mode=Validation")
            self.run_evaluation_epoch(self.dataloader["validation"],
                                      mode="validation", write_results=True)
        if self.eval_test:
            self.logger.info("Evaluating: mode=Test")
            if once_test: self.restart_dataloader("test")
            self.run_evaluation_epoch(self.dataloader["test"], mode="test",
                                      write_results=True)
        self.save()
        #self.log_tf_embeddings()
        self.pipeline.train()
        self.restart_dataloader("train")

    def get_rnn_hidden(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        hidden = self.pipeline.init_rnn_hidden(batch_size,
                                               self.train_on_gpu)
        return hidden

    def train(self, batch):
        self.pipeline.zero_grad()
        batch["inputs"] = batch["text"]
        batch["training"] = True
        batch["hidden"] = self.get_rnn_hidden()
        output = self.pipeline(batch)
        reg_loss = self.loss(output["reg_out"],
                         torch.unsqueeze(batch["sentiment"], dim=1))
        clf_loss = self.clf_loss(output["clf_out"],
                                 torch.max(batch["label"], 1)[1])
        loss = reg_loss + clf_loss
        scalar_loss = loss.data.cpu().numpy()[0] if self.train_on_gpu else loss.data.numpy()[0]
        scalar_reg_loss = reg_loss.data.cpu().numpy()[0] if self.train_on_gpu else \
                            reg_loss.data.numpy()[0]
        scalar_clf_loss = clf_loss.data.cpu().numpy()[0] if self.train_on_gpu \
                            else clf_loss.data.numpy()[0]
        loss.backward()
        self.optimizer.step()
        self.log("training.loss", loss)
        self.log("training.clf_loss", scalar_reg_loss)
        self.log("training.reg_loss", scalar_clf_loss)
        self.global_step += 1
        if self.global_step % 10 == 0:
            self.logger.info("Epoch: {}\tGlobal Step: {}\t"
                             "Training Loss: {}".format(self.current_epoch,
                                                self.global_step,
                                                scalar_loss))
            self.log_histogram()
            reg_pred = output["reg_out"].data.cpu().numpy() if \
                self.train_on_gpu else output["reg_out"].data.numpy()
            reg_gt = batch["sentiment"].data.cpu().numpy() if \
                self.train_on_gpu else batch["sentiment"].data.numpy()

            clf_pred = output["clf_out"].data.cpu().numpy() if \
                self.train_on_gpu else output["clf_out"].data.numpy()
            clf_gt = batch["label"].data.cpu().numpy() if \
                self.train_on_gpu else batch["label"].data.numpy()
            acc, f_score, clf_rpt = self._run_metrics(clf_pred, clf_gt)
            pc, p_val = self.metric(np.squeeze(reg_pred, axis=1), reg_gt)
            self.log("training.pearson_correlation", pc)
            self.log("training.pearson_correlation_p_val", p_val)
            self.log("training.Accuracy", acc)
            self.log("training.F1_Score", f_score)

    def _run_metrics(self, pred, gt):
        results = []
        for metric in self.clf_metrics:
            results.append(metric(pred, gt))
        return results[0], results[1], results[2]

    def run_evaluation_epoch(self, dataloader, mode="validation",
                             write_results=False, write_vectors=True,
                             verbose=False):
        val_losses, val_reg_losses, val_clf_losses, val_pcs, val_accs,\
        val_fscores = [], [], [], [], [], []
        for i_val, val_batch in dataloader:
            reg_pred, reg_gt, clf_pred, clf_gt, val_loss, val_reg_loss, \
            val_clf_loss, val_pc, val_p_val, acc, f_score, clf_rpt, output \
                = self.evaluate(val_batch)
            val_losses.append(val_loss)
            val_pcs.append(val_pc)
            val_accs.append(acc)
            val_fscores.append(f_score)
            val_reg_losses.append(val_reg_loss)
            val_clf_losses.append(val_clf_loss)
            if write_results:
                self.write_results_to_file(i_val, val_batch["text"], reg_pred,
                                           reg_gt, clf_pred, clf_gt, mode)
            if i_val % 10 == 0:
                if verbose:
                    self.logger.info("Epoch: {}\tGlobal Step: {}\t"
                                     "Mode: {}\tLoss: {}\tPC: {}\t"
                                     "Acc: {}\tF1: {}\tBatch "
                                     "Number: {}".format(
                            self.current_epoch, self.global_step, mode,
                            val_loss, val_pc, acc, f_score, i_val))
            if write_vectors:
                self.dump_output(output, i_val, mode)
        avg_loss, avg_pc, = np.average(val_losses), np.average(val_pcs)
        self.log("{}.average_loss".format(mode), avg_loss)
        avg_reg_loss = np.average(val_reg_losses)
        self.log("{}.Average_REG_Loss".format(mode), avg_reg_loss)
        self.log("{}.average_pearson_correlation".format(mode), avg_pc)
        avg_clf_loss, avg_acc, avg_f1score = np.average(val_clf_losses), \
                                         np.average(val_accs), \
                                         np.average(val_fscores)
        self.log("{}.Average_CLF_Loss".format(mode), avg_clf_loss)
        self.log("{}.Accuracy".format(mode), avg_acc)
        self.log("{}.F1_Score".format(mode), avg_f1score)
        if mode == "validation":
            self.update_loss_history(avg_loss)
        self.restart_dataloader(mode)

    def dump_output(self, output, batch_id, mode):

        attn_dir = os.path.join(
                self.experiment_config["{}_results".format(mode)],
                "attns")
        dump_dir = os.path.join(self.experiment_config["{}_results".format(mode)],
                                                       "dumps")
        if not os.path.exists(dump_dir): os.makedirs(dump_dir)
        if not os.path.exists(attn_dir): os.makedirs(attn_dir)

        file_path = os.path.join(dump_dir, "{}_epoch_{}_batch_{}.npy".format(mode,
                                                          self.current_epoch,
                                                          batch_id))
        out = output["3.RecurrentBlock.out"]
        out = out.data.cpu().numpy() if self.train_on_gpu else out.data.numpy()
        np.save(file_path, out)

    def evaluate(self, batch):
        batch["inputs"] = batch["text"]
        batch["training"] = False
        batch["hidden"] = self.get_rnn_hidden()
        output = self.pipeline(batch)
        reg_loss = self.loss(output["reg_out"],
                             torch.unsqueeze(batch["sentiment"], dim=1))
        clf_loss = self.clf_loss(output["clf_out"],
                                 torch.max(batch["label"], 1)[1])
        loss = reg_loss + clf_loss
        scalar_loss = loss.data.cpu().numpy()[0] if self.train_on_gpu else \
                        loss.data.numpy()[0]
        scalar_reg_loss = reg_loss.data.cpu().numpy()[0]\
                            if self.train_on_gpu else reg_loss.data.numpy()[0]
        scalar_clf_loss = clf_loss.data.cpu().numpy()[0]\
                            if self.train_on_gpu else clf_loss.data.numpy()[0]
        reg_pred = output["reg_out"].data.cpu().numpy() if \
            self.train_on_gpu else output["reg_out"].data.numpy()
        reg_gt = batch["sentiment"].data.cpu().numpy() if \
            self.train_on_gpu else batch["sentiment"].data.numpy()

        clf_pred = output["clf_out"].data.cpu().numpy() if \
            self.train_on_gpu else output["clf_out"].data.numpy()
        clf_gt = batch["label"].data.cpu().numpy() if \
            self.train_on_gpu else batch["label"].data.numpy()
        acc, f_score, clf_rpt = self._run_metrics(clf_pred, clf_gt)
        pc, p_val = self.metric(np.squeeze(reg_pred, axis=1), reg_gt)

        return reg_pred, reg_gt, clf_pred, clf_gt, scalar_loss, \
               scalar_reg_loss, scalar_clf_loss, pc, p_val, acc, f_score, \
               clf_rpt, output

    def load(self, model_path=None):
        model_path = model_path if model_path is not None else self.best_model_path
        self.logger.info("Trying to load checkpoint from '{}'".format(
                model_path))
        if os.path.exists(model_path):
            self.logger.info("Found checkpoint. Attemptimg to load it")
            checkpoint = torch.load(model_path)
            self.current_epoch = checkpoint['epoch'] if not self.fine_tuning \
                else 0
            self.global_step = checkpoint['global_step'] if not self.fine_tuning \
                else 0
            self.best_loss = checkpoint['best_loss']
            self.pipeline.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("Loaded checkpoint from: '{}'\n"
             " Current Epoch: {}\n Best Loss: {}".format(model_path,
                                 checkpoint['epoch'], checkpoint['best_loss']))
        else:
            self.logger.info(("Could not find checkpoint at : {} "
                              "Training fresh parameters".format(
                    model_path)))

    def save(self):
        if self.loss_has_improved() or self.eval_val == False:
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
        if len(self.loss_history) == 5:
            self.loss_history = self.loss_history[-4:]
        if not self.time_to_stop:
            if len(self.loss_history) > 0:
                delta = min(self.loss_history) - loss
                if delta >= self.early_stopping_delta:
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

    def write_results_to_file(self, batch_id, batch, pred, gt, clf_pred,
                              clf_gt, mode, tf_text=True):

        file_path = os.path.join(self.experiment_config["{}_results".format(
                mode)], "{}_epoch_{}_batch_{}.tsv".format(mode,
                                      self.current_epoch, batch_id))
        batch = batch.data.cpu().numpy().tolist() if self.train_on_gpu else \
                                            batch.data.numpy().tolist()
        text = self.sequence_decoder(batch)
        with codecs.open(file_path, "w", "utf-8") as rf:
            text_buff = []
            for t, p, g, clf_p, clf_g in zip(text, pred, gt, clf_pred, clf_gt):
                text_buff.append("{}\t{}\t{}\t{}\t{}".format(t, p, g, clf_p,
                                                             clf_g))
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
    elif dataset_config["name"].lower() == "amazon_reviews_imbalanced_de":
        ds = AmazonReviews(name="amazon_reviews_imbalanced_de",
                  lang="de", mode="regression",
                  use_spacy_vocab=dataset_config["use_spacy_vocab"],
                  load_spacy_vectors=dataset_config["load_spacy_vectors"],
                  spacy_model_name=dataset_config["spacy_model_name"],
                  aux=["word"], cuda=dataset_config["cuda"],
                  rescale=dataset_config["rescale"])
    elif dataset_config["name"].lower() == "stanford_sentiment_treebank":
        ds = SentimentTreebank(mode="regression",
                  use_spacy_vocab=dataset_config["use_spacy_vocab"],
                  load_spacy_vectors=dataset_config["load_spacy_vectors"],
                  spacy_model_name=dataset_config["spacy_model_name"],
                  aux=["word"], cuda=dataset_config["cuda"],
                  rescale=dataset_config["rescale"])
    elif dataset_config["name"].lower() == "amazon_reviews_balanced_de":
        ds = AmazonReviews(name="amazon_reviews_balanced_de",
                  lang="de", mode="regression",
                  use_spacy_vocab=dataset_config["use_spacy_vocab"],
                  load_spacy_vectors=dataset_config["load_spacy_vectors"],
                  spacy_model_name=dataset_config["spacy_model_name"],
                  aux=["word"], cuda=dataset_config["cuda"],
                  rescale=dataset_config["rescale"])
    elif dataset_config["name"].lower() == "kaggle_sentiment":
        ds = KaggleSentiment(mode="regression",
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
    pipeline = EmbeddingNgramCNNRNNMultiTask(vocab_size=vocab_size,
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
          regression_activation=pipeline_config["regression"]["activation"],
          cnn_dropout=pipeline_config["ngram_cnn"]["cnn_dropout"],
          rnn_dropout=pipeline_config["rnn"]["rnn_dropout"])
    return pipeline


def predict_pipeline(experiment_name, mode, dataset, setup, pipeline,
                 optimizer, experiment_config, _log, _run):
    ds = get_dataset_class(dataset_config=dataset)
    pipeline = get_pipeline(pipeline, vocab_size=ds.vocab_sizes["word"],
                            pretrained_word_vectors=ds.w2v, n_classes=ds.n_classes)
    trainer = KrishnaTrainer(experiment_name=experiment_name,
                                 pipeline=pipeline, dataset=ds,
                                 experiment_config=experiment_config,
                                 logger=_log,
                                 run=_run,
                                 n_epochs=setup["epochs"],
                                 batch_size=32,
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
                                 eval_test=setup["eval_test"],
                                 eval_val=setup["eval_val"],
                                 fine_tuning=setup["fine_tuning"])
    if mode == "train":
        trainer.run()
    elif mode == "predict":
        return trainer


def batchify(data, seq_max_len=0):
    if seq_max_len != 0:
        max_len = seq_max_len
    else:
        max_len = 0
        for val in data:
            max_len = len(val) if len(val) > max_len else max_len
    data = du.pad_sequences(
            data, padlen=max_len)
    return data


@ex.command
def predict(experiment_name, dataset, setup, pipeline,
                 optimizer, experiment_config, _log, _run):

    config = configs.THE_BOOK_OF_EXPERIMENTS["krishna"][
        "kaggle_sentiment"][0]
    config["setup"]["experiment_name"] = \
        "SA_EMBED_NGRAM_CNN_RNN_MULTITASK_krishna_kaggle_sentiment_0"
    config["experiment_name"] = config["setup"]["experiment_name"]
    dataset = {**dataset, **config["dataset"]}
    setup = {**setup, **config["setup"]}
    pipeline = {**pipeline, **config["pipeline"]}
    optimizer = {**optimizer, **config["optimizer"]}
    #experiment_config = experiment_boilerplate(setup)
    trainer = predict_pipeline(experiment_name, "predict", dataset, setup, pipeline,
                 optimizer, experiment_config, _log, _run)
    load_path = "/home/mindgarage07/.bhabana/experiments" \
                "/SA_EMBED_NGRAM_CNN_RNN_MULTITASK_krishna_kaggle_sentiment_0" \
                "/checkpoints/best_model.pth.tar"
    trainer.load(load_path)
    path = os.path.join(os.path.dirname(__file__), "kaggle_sentiment_test.tsv")
    with open(path, "r") as f, open("submission.txt", "w") as wf, \
            open("results.txt", "w") as rf:
        wf.write("PhraseId,Sentiment\n")
        rf.write("PhraseId,Sentiment\n")
        skip = True
        lines = f.readlines()
        batch, phrase_ids = [], []
        for line in tqdm(lines):
            if skip:
                skip = False
                continue
            cols = line.strip().split("\t")
            phrase_id = int(cols[0])
            text = cols[-1]
            phrase_ids.append(phrase_id)
            batch.append(trainer.preprocess(text))
            if len(batch) == 32:
                padded_batch = batchify(batch)
                reg_pred, clf_pred = trainer.predict(padded_batch, 32)
                cls = np.argmax(clf_pred, axis=1)
                for pid, c in zip(phrase_ids, cls):
                    wf.write("{},{}\n".format(pid, c))
                    rf.write("{},{}\n".format(pid, c))
                batch = []
                phrase_ids = []

        padded_batch = batchify(batch)
        reg_pred, clf_pred = trainer.predict(padded_batch, len(batch))
        cls = np.argmax(clf_pred, axis=1)
        for pid, c in zip(phrase_ids, cls):
            wf.write("{},{}\n".format(pid, c))
            rf.write("{},{}\n".format(pid, c))
        batch = []
        phrase_ids = []


@ex.automain
def run_pipeline(experiment_name, mode, dataset, setup, pipeline,
                 optimizer, experiment_config, _log, _run):
    ds = get_dataset_class(dataset_config=dataset)
    pipeline = get_pipeline(pipeline, vocab_size=ds.vocab_sizes["word"],
                            pretrained_word_vectors=ds.w2v, n_classes=ds.n_classes)
    trainer = KrishnaTrainer(experiment_name=experiment_name,
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
                                 eval_test=setup["eval_test"],
                                 eval_val=setup["eval_val"],
                                 load_path=setup["load_path"],
                                 fine_tuning=setup["fine_tuning"])
    if setup["mode"].lower() == "train":
        trainer.run()
    elif setup["mode"].lower() == "test":
        trainer.test()
    else:
        raise ValueError("Mode: {}, is not defined")