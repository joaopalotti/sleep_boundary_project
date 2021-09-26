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
from NN_commons import data_exists, load_data, save_data, create_xy
from NN_commons import run_tuning_procedure
from NN_commons import myXYDataset

import os
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

# +

class MyNet(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters()
        self.timestamp = datetime.now()
        self.labels = hparams.labels
        self.loss_fnct = hparams.loss_fnct
        self.weights = hparams.weights
        self.regression_tasks = hparams.regression_tasks
        self.classification_tasks = hparams.classification_tasks

        # Optimizer configs
        self.opt_learning_rate = hparams.opt_learning_rate
        self.opt_weight_decay = hparams.opt_weight_decay
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
        self.use_cnn = hparams.use_cnn
        self.cnn_kernel_size = hparams.cnn_kernel_size

        # Other configs
        self.batch_size = hparams.batch_size

        if self.use_cnn:
            self.cnn = nn.Conv1d(1, 1, kernel_size=self.cnn_kernel_size, stride=1, padding="same")

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



    # Options:
    # (1) input => feature extraction using TSFresh -> LSTM -> lin -> results
    # (2) input => no feature extraction -> LSTM -> lin -> results
    # (3) input => feature extraction with CNN (raw) -> LSTM = > LinearLayer = > Results


    def forward(self, x):
        if self.use_cnn:
            x = x.unsqueeze(1)  # Reshape from (B,L) to (B, C=1, L)
            x = self.cnn(x)
            x = x.squeeze(1)  # Reshape it back to (B,L)

        x = self.net(x)
        x = self.drop(x)
        out = {}
        for label in self.labels:
            out[label] = self.head[label](x)

        return out

    def configure_optimizers(self):
        print("Current number of parameters:", len(list(self.parameters())))
        optimizer = optim.Adam(self.parameters(), lr=self.opt_learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         patience=self.opt_step_size,
                                                         factor=self.opt_gamma,  # new_lr = lr * factor (default = 0.1)
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}

    def calculate_losses(self, y, predictions):

        y_label = {}
        for i, l in enumerate(self.labels):
            y_label[l] = y[:, i:i+1]

        losses = []
        for i, l in enumerate(self.labels):
            v = self.loss_fnct[i]()(predictions[l], y_label[l])
            losses.append(v)
            self.log('%s_loss' % l, v)

        final_loss = 0
        for i in range(len(self.labels)):
            final_loss = final_loss + (losses[i] * self.weights[i])

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
            y[l] = torch.stack([row["y"][0:, i:i+1] for row in outputs]).view(-1).cpu()
            pred[l] = torch.stack([row["preds"][l] for row in outputs]).view(-1)
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
            y[l] = torch.stack([row["y"][0:, i:i+1] for row in outputs]).view(-1).cpu()
            pred[l] = torch.stack([row["preds"][l] for row in outputs]).view(-1)
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


exp = "10min_centered"
datafolder = "../data/processed/train_test_splits/%s/" % exp
featset = "raw"  # Options are [raw, tsfresh]

if data_exists(datafolder, featset):
    print("Data already exist. Loading files from %s" % (datafolder))
    X, Y, test_pids = load_data(datafolder, featset)
else:
    X, Y, test_pids = create_xy(os.path.join(datafolder, "train_%s_data.csv.gz" % (featset)),
                                os.path.join(datafolder, "test_%s_data.csv.gz" % (featset)), use_gpu=True)
    save_data(datafolder, X, Y, test_pids, featset)


def do_parameter_tunning(mynet, datafolder, featset, ncpus=48, ngpus=3, ntrials=100, exp_name="test",
                         min_epochs=1, max_epochs=2):
    config = {
        # High level network configs
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        # Lower level details
        "bidirectional": tune.choice([True, False]),
        "lstm_layers": tune.choice([1, 2]),
        "hidden_dim": tune.choice([128, 64, 32, 16, 8]),
        "lstm_output_dim": tune.randint(8, 133),
        "dropout_lstm": tune.uniform(0, 1),
        "dropout_lin": tune.uniform(0, 1),
        # Optmizer
        "opt_step_size": tune.randint(1, 20),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        # Problem specific
        "labels": ["main_y"],
        "classification_tasks": ["main_y"],
        "regression_tasks": [],
        "loss_fnct": [nn.BCEWithLogitsLoss],
        "weights": [1.],
        # loss_fnct = [nn.BCEWithLogitsLoss, nn.L1Loss]
        # weight = [0.75, 0.25]
        "use_cnn": [True],
        "cnn_kernel_size": tune.randint(2, 11),
    }
    run_tuning_procedure(mynet, datafolder, featset, config, exp_name, ntrials=ntrials, ncpus=ncpus, ngpus=ngpus,
                         min_epochs=min_epochs, max_epochs=max_epochs)


# This needs to be the fullpath
do_parameter_tunning(MyNet,
                     "/export/sc2/jpalotti/github/sleep_boundary_project/data/processed/train_test_splits/%s/" % (exp),
                     featset, ncpus=48, ngpus=1, ntrials=50, # exp_name="exp_manyTo1_%s_%s" % (featset, exp),
                     exp_name="exp_cnnlstm_%s_%s" % (featset, exp),
                     min_epochs=1, max_epochs=50)

# +
# batch_size = 64
# dropout_lstm = 0.87986
# dropout_lin = 0.087821
# learning_rate = 0.00021999
# weight_decay = 0.00029587
# opt_step_size = 15
# hidden_dim = 128
# bidirectional = False
# lstm_layers = 2
# lstm_output_dim = 129
# use_cnn = True
# cnn_kernel_size = 5
#
# labels = ["main_y"] # , "percentage_y"]
# regression_tasks = [] # ["percentage_y"]
# classification_tasks = ["main_y"]
# loss_fnct = [nn.BCEWithLogitsLoss, nn.L1Loss]
# weights = [0.75, 0.25]
#
# hparams = Namespace(batch_size=batch_size,
#                     dropout_lstm=dropout_lstm,
#                     dropout_lin=dropout_lin,
#                     input_dim=X["train"].shape[1],
#                     use_cnn=use_cnn,
#                     cnn_kernel_size=cnn_kernel_size,
#                     #
#                     # Optmizer configs
#                     #
#                     opt_learning_rate=learning_rate,
#                     opt_weight_decay=weight_decay,
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
# best_parameters = {"batch_size": 1024, "bidirectional": True, "dropout_lin": 0.087821,
#                    "dropout_lstm": 0.87986, "hidden_dim": 128, "learning_rate": 0.00021999,
#                    "lstm_layers": 2, "lstm_output_dim": 129, "monitor": "loss",
#                    "opt_step_size": 15, "weight_decay": 0.00029587}
#
# results_MyNet_MP = eval_n_times(best_parameters, datafolder, 10, gpus=0, patience=3)
