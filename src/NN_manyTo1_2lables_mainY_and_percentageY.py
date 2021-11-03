# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from NN_commons import calculate_regression_metrics, calculate_classification_metrics, LSTMLayer
from NN_commons import run_tuning_procedure, eval_n_times
from data_commons import load_data

import os, sys
import ast
import pandas as pd
from argparse import Namespace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

wandb.login(key="e892ffdfa87d173880cbe0d6f831b7aff428c07c")

from ray import tune

from collections import OrderedDict
from datetime import datetime

os.environ["SLURM_JOB_NAME"] = "bash"


# +


def get_env_var(varname, default):
    return int(os.environ.get(varname)) if os.environ.get(varname) is not None else default

def chunks(l, n):
    n = len(l) // n
    return [l[i:i + n] for i in range(0, len(l), max(1, n))]


class MyNet(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters()
        self.timestamp = datetime.now()
        self.labels = hparams.labels
        self.loss_fnct = hparams.loss_fnct
        self.main_weight = hparams.main_weight
        self.regression_tasks = hparams.regression_tasks
        self.classification_tasks = hparams.classification_tasks

        # Optimizer configs
        self.opt_learning_rate = hparams.opt_learning_rate
        self.opt_step_size = hparams.opt_step_size
        self.opt_gamma = hparams.opt_gamma
        self.dropout_lstm = hparams.dropout_lstm
        self.dropout_lin = hparams.dropout_lin

        # LSTM Config
        self.hidden_dim = hparams.hidden_dim
        self.bidirectional = hparams.bidirectional
        self.lstm_layers = hparams.lstm_layers
        self.lstm_output_dim = hparams.lstm_output_dim
        self.input_dim = hparams.input_dim
        self.cnn_layers = hparams.cnn_layers
        self.cnn_kernel_size = hparams.cnn_kernel_size

        # Other configs
        self.batch_size = hparams.batch_size
        self.channels = 2
        # self.autoencoder = hparams.autoencoder

        if self.cnn_layers > 0:
            seqs = []
            for i in range(self.cnn_layers):
                kernel_size = self.cnn_kernel_size // (i + 1)
                kernel_size = 1 if kernel_size == 0 else kernel_size
                seq = nn.Sequential(nn.Conv1d(self.channels, self.channels,
                                              kernel_size=kernel_size, stride=1, padding="same"),
                                    nn.BatchNorm1d(self.channels),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.dropout_lin))
                seqs.append(seq)
            self.cnn = nn.Sequential(*seqs)

        self.net = LSTMLayer(input_size=self.input_dim, break_point=self.input_dim,
                             dropout_lstm=self.dropout_lstm,
                             dropout_lin=self.dropout_lin,
                             hidden_dim=self.hidden_dim,
                             bidirectional=self.bidirectional,
                             num_layers=self.lstm_layers,
                             output_dim=self.lstm_output_dim,
                             )

        self.drop = nn.Dropout(self.dropout_lin)

        self.head = nn.ModuleDict()
        for label in self.labels:
            self.head[label] = nn.Sequential(OrderedDict([
                ('lin1_%s' % label, nn.Linear(self.lstm_output_dim, 4)),
                ('act1_%s' % label, nn.ReLU(inplace=True)),
                ('dropout_%s' % label, nn.Dropout(self.dropout_lin)),
                ('lin2_%s' % label, nn.Linear(4, 1))
            ]))

        self.hs = None  # self.init_hidden()

    # Options:
    # (1) input => feature extraction using TSFresh -> LSTM -> lin -> results
    # (2) input => no feature extraction -> LSTM -> lin -> results
    # (3) input => feature extraction with CNN (raw) -> LSTM = > LinearLayer = > Results

    def init_hidden(self, batch_size):

        D = 2 if self.bidirectional is True else 1

        h0 = torch.zeros(self.lstm_layers * D, batch_size, self.hidden_dim).double()
        c0 = torch.zeros(self.lstm_layers * D, batch_size, self.hidden_dim).double()

        nn.init.xavier_normal_(h0)
        nn.init.xavier_normal_(c0)
        hidden = (h0, c0)

        return hidden

    def forward(self, x):

        if self.cnn_layers > 0:
            if self.channels == 2:
                x = x.view(x.shape[0], 2, x.shape[1] // 2)
                x = self.cnn(x)
            else:
                x = x.unsqueeze(1)  # Reshape from (B,L) to (B, C=1, L)
                x = self.cnn(x)
                x = x.squeeze(1)  # Reshape it back to (B,L)
            x = x.view(x.shape[0], -1)

        x, hidden = self.net(x)
        x = self.drop(x)
        out = {}
        for label in self.labels:
            out[label] = self.head[label](x)

        return out

    def configure_optimizers(self):
        print("Current number of parameters:", len(list(self.parameters())))
        optimizer = optim.AdamW(self.parameters(), lr=self.opt_learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         patience=self.opt_step_size,
                                                         factor=self.opt_gamma,  # new_lr = lr * factor (default = 0.1)
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}

    def calculate_losses(self, y, predictions):

        y_label = {}
        for i, l in enumerate(self.labels):
            y_label[l] = y[:, i:i + 1]

        losses = []
        for i, l in enumerate(self.labels):
            v = self.loss_fnct[i]()(predictions[l], y_label[l])
            losses.append(v)
            self.log('%s_loss' % l, v)

        final_loss = losses[0] * self.main_weight
        if len(self.labels) > 1:
            ave_weight = (1.0 - self.main_weight) / (len(self.labels) - 1)
            for i in range(1, len(self.labels)):
                final_loss += (losses[i] * ave_weight)

        return final_loss

    def process_step(self, batch):
        """
        batch is a list
        batch[0] contains features
        batch[1] contains both labels in one tensor
        """
        x, y = batch

        predictions = self(x)
        loss = self.calculate_losses(y, predictions)

        return predictions, y, loss

    def training_step(self, batch, batch_idx):
        predictions, y, loss = self.process_step(batch)
        self.log('loss', loss)
        return {'loss': loss, 'y': y, 'preds': predictions}

    def validation_step(self, batch, batch_idx):
        predictions, y, loss = self.process_step(batch)
        self.log('loss', loss)
        return {'loss': loss, 'y': y, 'preds': predictions}

    def test_step(self, batch, batch_idx):
        predictions, y, loss = self.process_step(batch)
        self.log('loss', loss)
        return {'loss': loss, 'y': y, 'preds': predictions}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([row['loss'] for row in outputs]).mean()
        print("(Validation) Total Loss: %.4f" % val_loss)

        y = {}
        pred = {}
        for i, l in enumerate(self.labels):
            y[l] = torch.cat([row["y"][0:, i:i+1] for row in outputs]).view(-1).cpu()
            pred[l] = torch.cat([row["preds"][l] for row in outputs]).view(-1)

            if l in self.classification_tasks:
                pred[l] = torch.round(torch.sigmoid(pred[l])).cpu()

        for label in self.classification_tasks:
            acc, prec, rec, f1, mcc = calculate_classification_metrics(y[label], pred[label])
            self.log("acc_%s" % label, acc)
            self.log("prec_%s" % label, prec)
            self.log("rec_%s" % label, rec)
            self.log("f1_%s" % label, f1)
            self.log("mcc_%s" % label, mcc)

            print("(Val_%s) Epoch: %d, Acc: %.3f, P: %.3f, R: %.3f, F1: %.3f, MCC: %.3f" % (label,
                                                                                            self.current_epoch,
                                                                                            acc, prec, rec, f1, mcc))

        for label in self.regression_tasks:
            MAE, MSE, r2 = calculate_regression_metrics(y[label], pred[label])

            self.log("MAE_%s" % label, MAE)
            self.log("MSE_%s" % label, MSE)
            self.log("r2_%s" % label, r2)

            print("(Val_%s) Epoch: %d, MAE: %.3f, MSE: %.3f, r2: %.3f" % (label, self.current_epoch, MAE, MSE, r2))

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([row['loss'] for row in outputs]).mean()
        print("(Test) Total Loss: %.4f" % test_loss)

        y = {}
        pred = {}
        for i, l in enumerate(self.labels):
            y[l] = torch.cat([row["y"][0:, i:i + 1] for row in outputs]).view(-1).cpu()
            pred[l] = torch.cat([row["preds"][l] for row in outputs]).view(-1)
            pred[l] = torch.round(torch.sigmoid(pred[l])).cpu()

        for label in self.classification_tasks:
            acc, prec, rec, f1, mcc = calculate_classification_metrics(y[label], pred[label])
            self.log("acc_%s" % label, acc)
            self.log("prec_%s" % label, prec)
            self.log("rec_%s" % label, rec)
            self.log("f1_%s" % label, f1)
            self.log("mcc_%s" % label, mcc)

            print("(Test_%s) Epoch: %d, Acc: %.3f, P: %.3f, R: %.3f, F1: %.3f, MCC: %.3f" % (label,
                                                                                             self.current_epoch,
                                                                                             acc, prec, rec, f1, mcc))

        for label in self.regression_tasks:
            MAE, MSE, r2 = calculate_regression_metrics(y[label], pred[label])

            self.log("MAE_%s" % label, MAE)
            self.log("MSE_%s" % label, MSE)
            self.log("r2_%s" % label, r2)

            print("(Test_%s) Epoch: %d, MAE: %.3f, MSE: %.3f, r2: %.3f" % (label, self.current_epoch, MAE, MSE, r2))


def do_parameter_tunning(mynet, datafolder, featset, config, ncpus=48, ngpus=3, ntrials=100, exp_name="test", patience=5,
                         min_epochs=1, max_epochs=2):
    output_file = "best_parameters_exp%s_trials%d.csv" % (exp_name, ntrials)
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    return run_tuning_procedure(mynet, datafolder, featset, config, exp_name, ntrials=ntrials, ncpus=ncpus, patience=patience,
                                ngpus=ngpus, min_epochs=min_epochs, max_epochs=max_epochs)


if __name__ == "__main__":

    NCPUS = get_env_var('SLURM_CPUS_PER_TASK', 12)
    NGPUS = 1
    NTRIALS = 50
    MIN_EPOCHS = 3
    MAX_EPOCHS = 30
    PATIENCE = 5
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)

    configs = {}
    configs["twoheads"] = {
        # High level network configs
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128, 256, 512, 1024, 2048]),
        # Lower level details
        "bidirectional": tune.choice([True, False]), "lstm_layers": tune.choice([1, 2]),
        "hidden_dim": tune.choice([128, 64, 32, 16, 8]), "lstm_output_dim": tune.randint(8, 256),
        "dropout_lstm": tune.uniform(0, 1), "dropout_lin": tune.uniform(0, 1),
        # Optmizer
        "opt_step_size": tune.randint(1, 20),
        # Version with two heads:
        "labels": ["main_y", "percentage_y"], "classification_tasks": ["main_y"], "regression_tasks": ["percentage_y"],
        "loss_fnct": [nn.BCEWithLogitsLoss, nn.MSELoss], "main_weight": tune.uniform(0, 1),
        #
        "cnn_layers": tune.randint(0, 4), "cnn_kernel_size": tune.randint(2, 11)
    }  # Loss function should be MCC -> double check it.

    configs["onehead"] = {
        # High level network configs
        "learning_rate": tune.loguniform(1e-6, 1e-1), "batch_size": tune.choice([64, 128, 256, 512, 1024, 2048]),
        # Lower level details
        "bidirectional": tune.choice([True, False]), "lstm_layers": tune.choice([1, 2]),
        "hidden_dim": tune.choice([128, 64, 32, 16, 8]), "lstm_output_dim": tune.randint(8, 256),
        "dropout_lstm": tune.uniform(0, 1), "dropout_lin": tune.uniform(0, 1),
        # Optmizer
        "opt_step_size": tune.randint(1, 20), "labels": ["main_y"],
        "classification_tasks": ["main_y"], "regression_tasks": [],
        "loss_fnct": [nn.BCEWithLogitsLoss], "main_weight": 1.,
        #
        "cnn_layers": tune.randint(0, 4), "cnn_kernel_size": tune.randint(2, 11),
    }

    configs["manyheads"] = {
        # High level network configs
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128, 256, 512, 1024, 2048]),
        # Lower level details
        "bidirectional": tune.choice([True, False]), "lstm_layers": tune.choice([1, 2]),
        "hidden_dim": tune.choice([128, 64, 32, 16, 8]), "lstm_output_dim": tune.randint(8, 256),
        "dropout_lstm": tune.uniform(0, 1), "dropout_lin": tune.uniform(0, 1),
        # Optmizer
        "opt_step_size": tune.randint(1, 20),
        # Version with two heads:
        "labels": ["main_y", "all_awake", "all_sleep", "is_transition", "percentage_y"],
        "classification_tasks": ["main_y", "all_awake", "all_sleep", "is_transition"],
        "regression_tasks": ["percentage_y"],
        "loss_fnct": [nn.BCEWithLogitsLoss, nn.BCEWithLogitsLoss, nn.BCEWithLogitsLoss, nn.BCEWithLogitsLoss, nn.MSELoss],
        "main_weight": tune.uniform(0, 1),
        "cnn_layers": tune.randint(0, 4), "cnn_kernel_size": tune.randint(2, 11)
    }  # Loss function should be MCC -> double check it.

    combinations = []
    for nheads in ["onehead", "twoheads", "manyheads"]:
        for win in ["10min_centered", "20min_centered", "40min_centered", "10min_notcentered",
                    "20min_notcentered", "40min_notcentered"]:
            for featset in ["tsfresh", "raw"]:
                combinations.append((nheads, win, featset))

    print("Total combinations:", len(combinations))
    print("All combinations:", combinations)
    selected_combinations = chunks(combinations, SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
    print("Processing: ", selected_combinations)

    for comb in selected_combinations:
        nheads, win, featset = comb
        config = configs[nheads]

        datafolder = "../data/processed/train_test_splits/%s/" % win
        exp_name = "exp_%s_%s_%s_n%d" % (nheads, featset, win, NTRIALS)
        print("Running experiment: %s" % exp_name)

        if os.path.exists("final_%s.csv" % exp_name):
            print("Experiment was already performed. Skipping %s" % exp_name)
            continue

        # This needs to be the fullpath
        best_df = do_parameter_tunning(MyNet,
                        "/export/sc2/jpalotti/github/sleep_boundary_project/data/processed/train_test_splits/%s/" % (win),
                        # "/home/palotti/github/sleep_boundary_project/data/processed/train_test_splits/%s/" % (win),
                        featset, config=config, ncpus=NCPUS, ngpus=NGPUS, ntrials=NTRIALS, exp_name=exp_name, patience=PATIENCE,
                        min_epochs=MIN_EPOCHS, max_epochs=MAX_EPOCHS)

        print("Done with parameter search")

        keys = [k for k in best_df.keys() if "config." in k]
        best_parameters = {}
        for k in keys:
            best_parameters[k.split("config.")[1]] = best_df[k].iloc[0]

        if nheads == "onehead":
            best_parameters["labels"] = ['main_y']
            best_parameters["classification_tasks"] = ['main_y']
            best_parameters["regression_tasks"] = []
            best_parameters["loss_fnct"] = [nn.BCEWithLogitsLoss]
            best_parameters["main_weight"] = 1.0
        elif nheads == "twoheads":
            best_parameters["labels"] = ["main_y", "percentage_y"]
            best_parameters["classification_tasks"] = ['main_y']
            best_parameters["regression_tasks"] = ["percentage_y"]
            best_parameters["loss_fnct"] = [nn.BCEWithLogitsLoss, nn.MSELoss]
        else:
            best_parameters["labels"] = ["main_y", "all_awake", "all_sleep", "is_transition", "percentage_y"]
            best_parameters["classification_tasks"] = ["main_y", "all_awake", "all_sleep", "is_transition"]
            best_parameters["regression_tasks"] = ["percentage_y"]
            best_parameters["loss_fnct"] = [nn.BCEWithLogitsLoss, nn.BCEWithLogitsLoss, nn.BCEWithLogitsLoss, nn.BCEWithLogitsLoss, nn.MSELoss]

        print("Final evaluation:")
        results_MyNet_MP = eval_n_times(MyNet, best_parameters, datafolder, featset, n=10, gpus=NGPUS,
                                        min_epochs=MIN_EPOCHS, max_epochs=MAX_EPOCHS,
                                        patience=PATIENCE, save_predictions=exp_name + "_pred.csv.gz")
        print(results_MyNet_MP)
        results_MyNet_MP["heads"] = nheads
        results_MyNet_MP["win"] = win
        results_MyNet_MP["featset"] = featset
        results_MyNet_MP["exp_name"] = exp_name
        results_MyNet_MP.to_csv("final_%s.csv" % exp_name)

        # best_parameters = {"labels": ["main_y", "percentage_y"],
        #                    "classification_tasks": ['main_y'],
        #                    "regression_tasks": ["percentage_y"],
        #                    "loss_fnct": [nn.BCEWithLogitsLoss, nn.L1Loss],
        #                    "main_weight": 1.0,
        #                    "learning_rate": 0.001,
        #                    "batch_size": 4048,
        #                    "bidirectional": True,
        #                    "lstm_layers": 1,
        #                    "hidden_dim": 16,
        #                    "lstm_output_dim": 8,
        #                    "dropout_lstm": 0.1,
        #                    "dropout_lin": 0.1,
        #                    "opt_step_size": 2,
        #                    "cnn_layers": 1,
        #                    "cnn_kernel_size": 3,
        #                    }
        #
        # results_MyNet_MP = eval_n_times(MyNet, best_parameters, datafolder, featset,
        #                                 n=1, gpus=0, patience=1, save_predictions=exp_name + "_pred.csv.gz")

# p = "batch_size=256,bidirectional=False,cnn_kernel_size=3,dropout_lin=0.50716,dropout_lstm=0.75856,hidden_dim=8,learning_rate=0.00067346,lstm_layers=1,lstm_output_dim=125,opt_step_size=13"
# best_parameters = dict([e.split("=") for e in p.split(",")])
# best_parameters = dict([(i, ast.literal_eval(v)) for i, v in best_parameters.items()])
#
# best_parameters["labels"] = ['main_y']
# best_parameters["classification_tasks"] = ['main_y']
# best_parameters["regression_tasks"] = []
# best_parameters["loss_fnct"] = [nn.BCEWithLogitsLoss]
# best_parameters["weights"] = [1.0]
#
# results_MyNet_MP = eval_n_times(MyNet, best_parameters, datafolder, featset, n=3, gpus=0, patience=3)
# print(results_MyNet_MP)
# results_MyNet_MP.to_csv("final_result_nn.csv")


# +
# batch_size = 64
# dropout_lstm = 0.87986
# dropout_lin = 0.087821
# learning_rate = 0.00021999
# opt_step_size = 15
# hidden_dim = 32
# bidirectional = True
# lstm_layers = 1
# lstm_output_dim = 32
# cnn_layers = 3
# cnn_kernel_size = 7
#
# labels = ["main_y", "percentage_y"]
# regression_tasks = ["percentage_y"]
# classification_tasks = ["main_y"]
# loss_fnct = [nn.BCEWithLogitsLoss, nn.L1Loss]
# weights = [0.75, 0.25]
#
# hparams = Namespace(batch_size=batch_size,
#                     dropout_lstm=dropout_lstm,
#                     dropout_lin=dropout_lin,
#                     input_dim=X["train"].shape[1],
#                     cnn_layers=cnn_layers,
#                     cnn_kernel_size=cnn_kernel_size,
#                     #
#                     # Optmizer configs
#                     #
#                     opt_learning_rate=learning_rate,
#                     opt_step_size=opt_step_size,
#                     opt_gamma=0.5,
#                     # LSTM configs
#                     hidden_dim=hidden_dim,
#                     bidirectional=bidirectional,
#                     lstm_layers=lstm_layers,
#                     lstm_output_dim=lstm_output_dim,
#                     #
#                     labels=labels,
#                     loss_fnct=loss_fnct,
#                     weights=weights,
#                     classification_tasks=classification_tasks,
#                     regression_tasks=regression_tasks
#                     )
#
#
# model = MyNet(hparams)
# model.double()
#
# train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True)
# val = DataLoader(myXYDataset(X["val"], Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True)
# test = DataLoader(myXYDataset(X["test"], Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True)
#
# # path_ckps = "./lightning_logs/test/"
# path_ckps = "./TEST_logs/lightning_logs/test/"
#
# wandb_logger = WandbLogger(name='Wandb_4')
#
# early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min', patience=5)
# ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}", save_top_k=1, verbose=False,
#                       monitor="loss", mode="min")
#
# trainer = Trainer(gpus=0, min_epochs=1, max_epochs=2, deterministic=True, callbacks=[early_stop_callback, ckp],
#                   logger=wandb_logger)
#
# trainer.fit(model, train, val)
# res = trainer.test(test_dataloaders=test)


# Run the same network several times
# p = "batch_size=256,bidirectional=False,cnn_kernel_size=3,dropout_lin=0.50716,dropout_lstm=0.75856,hidden_dim=8,learning_rate=0.00067346,lstm_layers=1,lstm_output_dim=125,opt_step_size=13"
# best_parameters = dict([e.split("=") for e in p.split(",")])
# best_parameters = dict([(i, ast.literal_eval(v)) for i, v in best_parameters.items()])
#
# best_parameters["labels"] = ['main_y']
# best_parameters["classification_tasks"] = ['main_y']
# best_parameters["regression_tasks"] = []
# best_parameters["loss_fnct"] = [nn.BCEWithLogitsLoss]
# best_parameters["weights"] = [1.0]
# best_parameters["cnn_layers"] = True
#
# results_MyNet_MP = eval_n_times(MyNet, best_parameters, datafolder, featset, n=3, gpus=0, patience=3)
# print(results_MyNet_MP)
# results_MyNet_MP.to_csv("final_result_nn.csv")
