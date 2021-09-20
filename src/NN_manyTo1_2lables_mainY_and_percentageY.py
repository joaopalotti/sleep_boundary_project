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

from src.NN_commons import calculate_regression_metrics, calculate_classification_metrics, LSTMLayer
from src.NN_commons import data_exists, load_data, save_data, create_xy
from src.NN_commons import run_tuning_procedure
from src.NN_commons import myXYDataset

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

        # Other configs
        self.batch_size = hparams.batch_size

        self.net = LSTMLayer(input_size=self.input_dim, break_point=self.input_dim,
                             dropout_lstm=self.dropout_lstm,
                             dropout_lin=self.dropout_lin,
                             hidden_dim=self.hidden_dim,
                             bidirectional=self.bidirectional,
                             num_layers=self.lstm_layers,
                             output_dim=self.lstm_output_dim,
                             )

        self.drop = nn.Dropout(self.dropout_lin)
        self.head = nn.ModuleDict({
            'main_y': nn.Sequential(OrderedDict([
                ('lin1', nn.Linear(self.lstm_output_dim, 4)),
                ('act1', nn.ReLU(inplace=True)),
                ('dropout', nn.Dropout(self.dropout_lin)),
                ('lin2', nn.Linear(4, 1))
            ])),

            'percentage_y': nn.Sequential(OrderedDict([
                ('lin1', nn.Linear(self.lstm_output_dim, 4)),
                ('act1', nn.ReLU(inplace=True)),
                #             ('act1', nn.ReLU(inplace=False)),
                ('dropout', nn.Dropout(self.dropout_lin)),
                ('lin2', nn.Linear(4, 1))
            ])),
        })

    def forward(self, x):
        x = self.net(x)
        x = self.drop(x)
        out = {}
        out['main_y'] = self.head['main_y'](x)
        out['percentage_y'] = self.head['percentage_y'](x)

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
        classificationl_loss_fnct = nn.BCEWithLogitsLoss()
        regression_loss_fnct = nn.L1Loss()

        label = {}
        label['main_y'] = y[:, :1]
        label['percentage_y'] = y[:, 1:2]

        closs = classificationl_loss_fnct(predictions["main_y"], label["main_y"])
        rloss = regression_loss_fnct(predictions["percentage_y"], label["percentage_y"])

        self.log('mainY_loss', closs)
        self.log('percentageY_loss', rloss)

        final_loss = 0.75 * closs + 0.25 * rloss

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
        y['main_y'] = torch.stack([row["y"][0:, 0:-1] for row in outputs]).view(-1).cpu()
        y['percentage_y'] = torch.stack([row["y"][0:, 1:] for row in outputs]).view(-1).cpu()

        pred = {}
        pred['main_y'] = torch.stack([row["preds"]['main_y'] for row in outputs]).view(-1)
        pred['percentage_y'] = torch.stack([row["preds"]['percentage_y'] for row in outputs]).view(-1)

        pred['main_y'] = torch.round(torch.sigmoid(pred['main_y']))
        pred['main_y'] = pred['main_y'].cpu()
        pred['percentage_y'] = pred['percentage_y'].cpu()

        key_list = ['main_y', 'percentage_y']

        acc_main_y, prec_main_y, rec_main_y, f1_main_y, mcc_main_y = calculate_classification_metrics(y['main_y'],
                                                                                                      pred['main_y'])

        self.log("acc_main_y", acc_main_y)
        self.log("prec_main_y", prec_main_y)
        self.log("rec_main_y", rec_main_y)
        self.log("f1_main_y", f1_main_y)
        self.log("mcc_main_y", mcc_main_y)

        print("(Val_main_y) Epoch: %d, Acc: %.3f, P: %.3f, R: %.3f, F1: %.3f, MCC: %.3f" % (self.current_epoch,
                                                                                            acc_main_y, prec_main_y,
                                                                                            rec_main_y, f1_main_y,
                                                                                            mcc_main_y))

        MAE_y, MSE_y, r2_y = calculate_regression_metrics(y['percentage_y'], pred['percentage_y'])

        self.log("MAE_y", MAE_y)
        self.log("MSE_y", MSE_y)
        self.log("r2_y", r2_y)

        print("(Val_percentage_y) Epoch: %d, MAE_y: %.3f, MSE_y: %.3f, r2: %.3f" % (self.current_epoch,
                                                                                    MAE_y, MSE_y, r2_y))

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([row['loss'] for row in outputs]).mean()
        print("(Test) Total Loss: %.4f" % test_loss)

        y = {}
        y['main_y'] = torch.stack([row["y"][0:, 0:-1] for row in outputs]).view(-1).cpu()
        y['percentage_y'] = torch.stack([row["y"][0:, 1:] for row in outputs]).view(-1).cpu()

        pred = {}
        pred['main_y'] = torch.stack([row["preds"]['main_y'] for row in outputs]).view(-1)
        pred['percentage_y'] = torch.stack([row["preds"]['percentage_y'] for row in outputs]).view(-1)
        pred['main_y'] = torch.round(torch.sigmoid(pred['main_y']))
        pred['main_y'] = pred['main_y'].cpu()
        pred['percentage_y'] = pred['percentage_y'].cpu()

        acc_main_y, prec_main_y, rec_main_y, f1_main_y, mcc_main_y = calculate_classification_metrics(y['main_y'],
                                                                                                      pred['main_y'])

        self.log("acc_main_y", acc_main_y)
        self.log("prec_main_y", prec_main_y)
        self.log("rec_main_y", rec_main_y)
        self.log("f1_main_y", f1_main_y)
        self.log("mcc_main_y", mcc_main_y)
        print("TEST: Acc: %.3f, P: %.3f, R: %.3f, F1: %.3f, MCC: %.3f" % (
            acc_main_y, prec_main_y, rec_main_y, f1_main_y, mcc_main_y))

        MAE_y, MSE_y, r2_y = calculate_regression_metrics(y['percentage_y'], pred['percentage_y'])

        self.log("MAE_y", MAE_y)
        self.log("MSE_y", MSE_y)
        self.log("r2_y", r2_y)

        print("(TEST: MAE_y: %.3f, MSE_y: %.3f, r2: %.3f" % (MAE_y, MSE_y, r2_y))


datafolder = "../data/processed/train_test_splits/10min_centered/"
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
    }
    run_tuning_procedure(mynet, datafolder, featset, config, exp_name, ntrials=ntrials, ncpus=ncpus, ngpus=ngpus,
                         min_epochs=min_epochs, max_epochs=max_epochs)


# This needs to be the fullpath
do_parameter_tunning(MyNet,
                     "/home/palotti/github/sleep_boundary_project/data/processed/train_test_splits/10min_centered/",
                     featset, ncpus=48, ngpus=1, ntrials=50, exp_name="exp_manyTo1_%s" % (featset),
                     min_epochs=1, max_epochs=50)

# +
# batch_size = 256
# dropout_lstm = 0.87986
# dropout_lin = 0.087821
# learning_rate = 0.00021999
# weight_decay = 0.00029587
# opt_step_size = 15
# hidden_dim = 128
# bidirectional = False
# lstm_layers = 2
# lstm_output_dim = 129
#
# hparams = Namespace(batch_size=batch_size,
#                     dropout_lstm=dropout_lstm,
#                     dropout_lin=dropout_lin,
#                     input_dim=X["train"].shape[1],
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
