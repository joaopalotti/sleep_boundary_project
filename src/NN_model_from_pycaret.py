# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
pd.set_option('display.max_rows', 20)

import numpy as np
import os
from tqdm import tqdm
from glob import glob

from pycaret.classification import *
from sklearn.model_selection import KFold

from argparse import Namespace
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only, seed

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
    
from collections import OrderedDict
from datetime import datetime

# -

def pycater_setup(train_data, test_data, 
                  gt_label = "label_5min",
                  ignore_feat= ["id", "fold", "linetime", "activity"],
                  use_gpu=False):
    
    experiment = setup(data=train_data, test_data=test_data,
                       target=gt_label, session_id=123,
                       normalize=True, transformation=True,
                       fold_strategy="groupkfold", fold_groups="fold",
                       ignore_features= ignore_feat,
                       silent=True, use_gpu=use_gpu,
                       # normalize_method = 'zscore',
                       normalize_method = 'minmax',       
                       # remove_outliers = True,
                       polynomial_features = True,
                       # fix_imbalance = True,
                   )
    return experiment


# +
def data_exists(datafolder, suffix="pycaret"):
    if not os.path.exists(datafolder):
        return False
    
    for f in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]: 
        if not os.path.exists("%s/%s_%s.csv.gz" % (datafolder, f, suffix)):
            return False
        
    return True

def save_data(output_folder, X, Y):
    X["train"].to_csv("%s/X_train_pycaret.csv.gz" % (output_folder), index=False)
    X["val"].to_csv("%s/X_val_pycaret.csv.gz" % (output_folder), index=False)
    X["test"].to_csv("%s/X_test_pycaret.csv.gz" % (output_folder), index=False)

    Y["train"].to_csv("%s/y_train_pycaret.csv.gz" % (output_folder), index=False)
    Y["val"].to_csv("%s/y_val_pycaret.csv.gz" % (output_folder), index=False)
    Y["test"].to_csv("%s/y_test_pycaret.csv.gz" % (output_folder), index=False)

def load_data(datafolder):
    X, Y = {}, {}
    X["train"] = pd.read_csv("%s/X_train_pycaret.csv.gz" % (datafolder))
    X["val"] = pd.read_csv("%s/X_val_pycaret.csv.gz" % (datafolder))
    X["test"] = pd.read_csv("%s/X_test_pycaret.csv.gz" % (datafolder))
    
    Y["train"] = pd.read_csv("%s/y_train_pycaret.csv.gz" % (datafolder))
    Y["val"] = pd.read_csv("%s/y_val_pycaret.csv.gz" % (datafolder))
    Y["test"] = pd.read_csv("%s/y_test_pycaret.csv.gz" % (datafolder))
    
    return X, Y


# -

def extract_features(train_path="train_data.csv.gz", test_path="test_data.csv.gz"):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    experiment = pycater_setup(train_data, test_data, 
                               gt_label = "ground_truth", ignore_feat = ["pid", "fold"])

    # Now we extract the postprocessed features
    # and append back the fold numbers to further break it into train/val
    X_train = get_config("X_train")
    X_test = get_config("X_test")

    y_train = get_config("y_train")
    y_test = get_config("y_test")

    X_train = pd.concat([train_data["fold"], X_train], axis=1)
    y_train = pd.concat([train_data["fold"], y_train], axis=1)
    
    # Data is well distributed across the folds
    for f in X_train["fold"].unique():
        print(f, X_train[X_train["fold"] == f].shape)
        
    X_val = X_train[X_train["fold"] == 9]
    X_train = X_train[X_train["fold"] != 9]

    y_val = y_train[y_train["fold"] == 9]
    y_train = y_train[y_train["fold"] != 9]

    del X_train["fold"], X_val["fold"], y_val["fold"], y_train["fold"]
    
    X = {"train": X_train, "val": X_val, "test": X_test}
    Y = {"train": Y_train, "val": Y_val, "test": Y_test}
    
    return X, Y


# +
class myXYDataset(Dataset):
    def __init__(self, X, Y):
        self.Y = Y
        self.X = X

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X.iloc[idx].values.astype(np.double)
        y = self.Y.iloc[idx].values.astype(np.double)
        return x, y
    
def calculate_classification_metrics(labels, predictions):
    return metrics.accuracy_score(labels, predictions),\
            metrics.precision_score(labels, predictions), \
            metrics.recall_score(labels, predictions), \
            metrics.f1_score(labels, predictions, average='micro'), \
            metrics.matthews_corrcoef(labels, predictions)


# +
    
def get_number_internal_layers(n, output_size):
    """
    E.g.:
        get_number_internal_layers(20, 3) --> [16, 8, 4]
        get_number_internal_layers(192, 16) # --> [128, 64, 32]
    """
    i = 1; d = 2; s = []
    while (n - 1) / d > 1: 
        s.append(d)
        i += 1
        d = 2**i;
        
    s = [e for e in s if e > output_size]
    return s[::-1]


class LSTMLayer(pl.LightningModule):
    def __init__(self,
                 input_size=8,
                 hidden_dim=10,
                 output_dim=2,
                 dropout_lstm=0.0,
                 dropout_lin=0.0,
                 break_point=None,
                 bidirectional=False,
                 num_layers=1,
                 ):
        super(LSTMLayer, self).__init__()

        if break_point is None:
            break_point = input_size
        
        print("BREAK POINT:", break_point)
        
        
        self.lstm = nn.LSTM(break_point, hidden_dim, num_layers=num_layers, dropout=dropout_lstm,
                            batch_first=True, bidirectional=bidirectional)
        self.linlayers = nn.ModuleList()
        self.drop = nn.Dropout(dropout_lin)
        
        if bidirectional:
            hidden_dim *= 2
        
        last_d = hidden_dim * (input_size//break_point)
        for lay_size in get_number_internal_layers(last_d, output_dim):
            print("Last: %d, Next: %d" % (last_d, lay_size))
            self.linlayers.append(nn.Sequential(nn.Linear(last_d, lay_size), 
                                                nn.ReLU(inplace=True), 
                                                nn.Dropout(dropout_lin))
                                 )
            last_d = lay_size
            
        print("Very Last: %d, Out: %d" % (last_d, output_dim))
        print("#Lin layers: ", len(self.linlayers))
        
        self.last_lin  = nn.Sequential(nn.Linear(last_d, output_dim), nn.ReLU(inplace=True))
        self.break_point = break_point
        
    def forward(self, x):
#         print("INPUT:", x.shape)
#         print("BP:", self.break_point)
#         print("second dim:", x.shape[1]//self.break_point)
        x = x.view(x.shape[0], x.shape[1]//self.break_point, -1)
#         print("Reshaped to:", x.shape)
        
        x, hidden = self.lstm(x)
#         print("After LSTM:", x.shape)
        
        x = x.reshape(x.shape[0], -1)
        
#         print("After reshape:", x.shape)
        for lay in self.linlayers:
            x = lay(x)
        
        x = self.last_lin(x)
        return x


# -

class MyNet(pl.LightningModule):

    def __init__(self, hparams):

        super().__init__()

        self.save_hyperparameters()
        self.hparams = hparams
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
        

        # Other configs
        self.batch_size = hparams.batch_size
        
        # self.net = nn.Sequential(nn.Linear(113, 64), nn.ReLU(inplace=True))
        self.net = LSTMLayer(input_size=113, break_point=113,
                             dropout_lstm=self.dropout_lstm,
                             dropout_lin=self.dropout_lin,
                             hidden_dim=self.hidden_dim,
                             bidirectional=self.bidirectional,
                             num_layers=self.lstm_layers,
                             output_dim=self.lstm_output_dim,
                            )
        
        self.drop = nn.Dropout(self.dropout_lin)
        self.head = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(self.lstm_output_dim, 4)),
            ('act1', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(self.dropout_lin)),
            ('lin2', nn.Linear(4, 1))
        ]))

        
    def forward(self, x):
        # Probably we want to use the hidden state here
        x = self.net(x)
        x = self.drop(x)
        out = self.head(x)
        return out

    def configure_optimizers(self):
        print("Current number of parameters:", len(list(self.parameters())))
        optimizer = optim.Adam(self.parameters(), lr=self.opt_learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min',
                                                         patience=self.opt_step_size,
                                                         factor=self.opt_gamma, # new_lr = lr * factor (default = 0.1)
                                                         verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}

    def calculate_losses(self, y, predictions):
        loss_fnct = nn.BCEWithLogitsLoss()
        loss = loss_fnct(predictions, y)
        return loss

    def process_step(self, batch):
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

        y = torch.stack([row["y"] for row in outputs]).view(-1).cpu()
        pred = torch.stack([row['preds'].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()
            
        acc, prec, rec, f1, mcc = calculate_classification_metrics(y, pred)
        self.log("acc", acc)
        self.log("prec", prec)
        self.log("rec", rec)
        self.log("f1", f1)
        self.log("mcc", mcc)
        
        print("(Val) Epoch: %d, Acc: %.3f, P: %.3f, R: %.3f, F1: %.3f, MCC: %.3f" % (self.current_epoch,
                                                                                     acc, prec, rec, f1, mcc))
        
    def test_epoch_end(self, outputs):
        test_loss = torch.stack([row['loss'] for row in outputs]).mean()
        print("(Test) Total Loss: %.4f" % test_loss)

        y = torch.stack([row["y"] for row in outputs]).view(-1).cpu()
        pred = torch.stack([row['preds'].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()
            
        acc, prec, rec, f1, mcc = calculate_classification_metrics(y, pred)
        self.log("acc", acc)
        self.log("prec", prec)
        self.log("rec", rec)
        self.log("f1", f1)
        self.log("mcc", mcc)
        print("TEST: Acc: %.3f, P: %.3f, R: %.3f, F1: %.3f, MCC: %.3f" % (acc, prec, rec, f1, mcc))


# +
# datafolder = "../data/processed_pycaret_5min/."

# if data_exists(datafolder):
#     X, Y = load_data(datafolder)
# else:
#     X, Y = extract_features("train_data.csv.gz", "test_data.csv.gz")
#     save_data(datafolder, X, Y)

# +
# # # THis cell runs the network one time. 
# # # We can see that we need parameter tunning

# batch_size = 2024
# dropout_lstm = 0.00
# dropout_lin = 0.00
# learning_rate = 0.01
# weight_decay = 0.01
# opt_step_size = 10
# hidden_dim = 64
# bidirectional = True
# lstm_layers = 1
# lstm_output_dim=16

# hparams = Namespace(batch_size=batch_size,
#                     dropout_lstm=dropout_lstm,
#                     dropout_lin=dropout_lin,
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

# model = MyNet(hparams) 
# model.double()

# train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
# val   = DataLoader(myXYDataset(X["val"],   Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
# test  = DataLoader(myXYDataset(X["test"],  Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)

# path_ckps = "./lightning_logs/test/"

# early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min', patience=5)
# ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}", save_top_k=1, verbose=False, 
#                       prefix="", monitor="loss", mode="min")

# trainer = Trainer(gpus=0, min_epochs=1, max_epochs=2, callbacks=[early_stop_callback, ckp])
# trainer.fit(model, train, val)
# res = trainer.test(test_dataloaders=test)

# -
def hyper_tuner(config, datafolder = "/home/palotti/github/sleep_boundary_project/data/processed_pycaret_5min/"):
    
    monitor = config["monitor"] # What to monitor? MCC/F1 or loss?
    
    # High level network configs
    batch_size = config["batch_size"]
    
    # Lower level details
    bidirectional = config["bidirectional"]
    hidden_dim = config["hidden_dim"]
    lstm_layers = config["lstm_layers"]
    dropout_lstm = config["dropout_lstm"]
    dropout_lin = config["dropout_lin"]
    lstm_output_dim = config["lstm_output_dim"]
    
    # Optmizer
    learning_rate = config["learning_rate"]
    opt_step_size = config["opt_step_size"]
    weight_decay = config["weight_decay"]
    
    X, Y = load_data(datafolder)
    
    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val   = DataLoader(myXYDataset(X["val"],   Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    test  = DataLoader(myXYDataset(X["test"],  Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    
    seed.seed_everything(42)

    path_ckps = "./lightning_logs/test/"

    if monitor == "mcc":
        early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='mcc', mode='max', patience=3)
        ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False, prefix="",
                              monitor="mcc", mode="max")
    else:
        early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min', patience=3)
        ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False, prefix="",
                              monitor="loss", mode="min")
    
    hparams = Namespace(batch_size=batch_size,
                        #
                        # Optmizer configs
                        #
                        opt_learning_rate=learning_rate,
                        opt_weight_decay=weight_decay,
                        opt_step_size=opt_step_size,
                        opt_gamma=0.5,
                        # LSTM configs
                        hidden_dim=hidden_dim,
                        bidirectional=bidirectional,
                        lstm_layers=lstm_layers,
                        lstm_output_dim=lstm_output_dim,
                        dropout_lstm=dropout_lstm,
                        dropout_lin=dropout_lin,
                       )

    model = MyNet(hparams)
    model.double()
    
    tune_metrics = {"loss": "loss", "mcc": "mcc", "acc": "acc", "prec": "prec", "rec": "rec", "f1": "f1"}
    tune_cb = TuneReportCallback(tune_metrics, on="validation_end")
    
    trainer = Trainer(gpus=0, min_epochs=2, max_epochs=5,
                      callbacks=[early_stop_callback, ckp, tune_cb])
    trainer.fit(model, train, val)


def run_tuning_procedure(config, expname, ntrials, ncpus, ngpus):

    trainable = tune.with_parameters(hyper_tuner)

    analysis = tune.run(trainable,
                        resources_per_trial={"cpu": ncpus, "gpu": ngpus},
                        metric="loss",
                        mode="min",
                        config=config,
                        num_samples=ntrials,
                        name=expname)

    print("Best Parameters:", analysis.best_config)

    analysis.best_result_df.to_csv("best_parameters_exp%s_trials%d.csv" % (expname, ntrials))
    analysis.results_df.to_csv("all_results_exp%s_trials%d.csv" % (expname, ntrials))
    print("Best 5 results")
    print(analysis.results_df.sort_values(by="mcc", ascending=False).head(5))



# +
config_lstm = {
    # What to monitor? MCC/F1 or loss?
    "monitor": tune.choice(["loss", "mcc"]),  
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

ncpus=12
ngpus=0
ntrials=2
exp_name = "test"

run_tuning_procedure(config_lstm, exp_name, ntrials=ntrials, ncpus=ncpus, ngpus=ngpus)

