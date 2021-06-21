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


# +
# train_raw_dataa = pd.read_csv("../data/files/train_raw_data.csv.gz")
# test_raw_dataa = pd.read_csv("../data/files/test_raw_data.csv.gz")
# # # train_raw_dataa.pid.unique()     
# good = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
# test_list = [110, 111, 144, 149, 220]
# new_df_train = train_raw_dataa[train_raw_dataa.pid.isin(good)]
# new_df_test = test_raw_dataa[test_raw_dataa.pid.isin(test_list)]

# new_df_train.to_csv('../data/files/part of raw data/train_raw_data.csv.gz')
# new_df_test.to_csv('../data/files/part of raw data/test_raw_data.csv.gz')
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

def save_data(output_folder, X, Y, test_pids):
    X["train"].to_csv("%s/X_train_pycaret.csv.gz" % (output_folder), index=False)
    X["val"].to_csv("%s/X_val_pycaret.csv.gz" % (output_folder), index=False)
    X["test"].to_csv("%s/X_test_pycaret.csv.gz" % (output_folder), index=False)

    Y["train"].to_csv("%s/y_train_pycaret.csv.gz" % (output_folder), index=False)
    Y["val"].to_csv("%s/y_val_pycaret.csv.gz" % (output_folder), index=False)
    Y["test"].to_csv("%s/y_test_pycaret.csv.gz" % (output_folder), index=False)

    test_pids.to_csv("%s/test_pids.csv.gz" % (output_folder))
    
def load_data(datafolder):
    X, Y = {}, {}
    X["train"] = pd.read_csv("%s/X_train_pycaret.csv.gz" % (datafolder))
    X["val"] = pd.read_csv("%s/X_val_pycaret.csv.gz" % (datafolder))
    X["test"] = pd.read_csv("%s/X_test_pycaret.csv.gz" % (datafolder))
    
    Y["train"] = pd.read_csv("%s/y_train_pycaret.csv.gz" % (datafolder))
    Y["val"] = pd.read_csv("%s/y_val_pycaret.csv.gz" % (datafolder))
    Y["test"] = pd.read_csv("%s/y_test_pycaret.csv.gz" % (datafolder))
    
    test_pids = pd.read_csv("%s/test_pids.csv.gz" % (datafolder))
    
    return X, Y, test_pids


# +
def extract_features(path_to_raw_features="../data/files/", 
                    signals = ["activity", "mean_hr"]):
    
# def extract_features(train_path="../data/files/part of raw data/train_data.csv.gz", test_path="../data/files/part of raw data/test_data.csv.gz", use_gpu=False):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    test_pids = test_data[["pid", "gt_time"]]
    
    experiment = pycater_setup(train_data, test_data, 
                               gt_label = "ground_truth", ignore_feat = ["pid", "fold", "gt_time"],
                               use_gpu=use_gpu)

    # Now we extract the postprocessed features
    # and append back the fold numbers to further break it into train/val
    X_train = get_config("X_train")
    X_test = get_config("X_test")

    y_train = get_config("y_train")
    y_test = get_config("y_test")

    if isinstance(y_test, pd.Series):
        y_test = y_test.to_frame()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame()
    
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
    Y = {"train": y_train, "val": y_val, "test": y_test}
    
    return X, Y, test_pids


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
            metrics.f1_score(labels, predictions, average='weighted'), \
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
        self.hidden_dim = hidden_dim
        
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
            
            
        # define hidden state and cell state
        
        if bidirectional:
#         self.h0 = torch.rand(num_layers, num_layers*hidden_dim, hidden_dim)
            self.h0 = torch.zeros(num_layers*2, hidden_dim, self.hidden_dim)
            self.c0 = torch.zeros(num_layers*2, hidden_dim, self.hidden_dim)
        else:
            self.h0 = torch.zeros(num_layers, num_layers*hidden_dim, hidden_dim)
            self.c0 = torch.zeros(num_layers, num_layers*hidden_dim, hidden_dim)


        print("Very Last: %d, Out: %d" % (last_d, output_dim))
        print("#Lin layers: ", len(self.linlayers))
        
        self.last_lin  = nn.Sequential(nn.Linear(last_d, output_dim), nn.ReLU(inplace=True))
        self.break_point = break_point
        
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        
        
    def forward(self, x):
        print("type x", type(x))
        print("INPUT x is:", x.shape)
        print("BP:", self.break_point)

        print("second dim:", x.shape[1]//self.break_point)
        x = x.view(x.shape[0], x.shape[1]//self.break_point, -1)
        print("INPUT x is:", x.shape)
        print("after using viw with x size 0", x.size(0))
        

        hn = torch.rand(self.num_layers, x.shape[0], self.output_dim)
        cn = torch.zeros(self.num_layers, x.shape[0], self.output_dim)
        
        # xavier for initializing h0
        nn.init.xavier_normal_(self.h0)

        # TO GET RID OF AN ERROR, CONVERT IT TO DOUBEL
        self.h0 = self.h0.double()
        self.c0 = self.c0.double()
        

        x, (hn, cn) = self.lstm(x, (self.h0, self.c0))
    
        x = x.reshape(x.shape[0], -1)
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
        #input_size and break_point should be the same values(number of features or x[1])
        self.net = LSTMLayer(input_size=90, break_point=90,
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
        pred = torch.stack([row['preds'] for row in outputs]).view(-1)
        pred = torch.round(torch.sigmoid(pred))
        pred = pred.cpu()
            
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
        pred = torch.stack([row['preds'] for row in outputs]).view(-1)
        pred = torch.round(torch.sigmoid(pred))
        pred = pred.cpu()
            
        acc, prec, rec, f1, mcc = calculate_classification_metrics(y, pred)
        self.log("acc", acc)
        self.log("prec", prec)
        self.log("rec", rec)
        self.log("f1", f1)
        self.log("mcc", mcc)
        print("TEST: Acc: %.3f, P: %.3f, R: %.3f, F1: %.3f, MCC: %.3f" % (acc, prec, rec, f1, mcc))

train

# ### The next two cells are able to run the network one single time. 
# It is useful for us to debug the network before running the param tuning

# +
datafolder = "../data/files/part of raw data/processed_pycaret_5min/."
# datafolder = "../data/files/part of raw data/processed_pycaret_5min/"

if data_exists(datafolder):
    X, Y, test_pids = load_data(datafolder)
    #this part works and it doesn't run the 'else'
else:
    X, Y, test_pids = extract_features("train_data.csv.gz", "test_data.csv.gz", use_gpu=True)
#     X, Y, test_pids = extract_features("../data/files/part of raw data/train_raw_data.csv.gz", "../data/files/part of raw data/test_raw_data.csv.gz", use_gpu=True)
    save_data(datafolder, X, Y, test_pids)
    
    


# +
# model
# -

X['train'].shape[0]

# +
batch_size = 1024
# batch_size = 256
dropout_lstm = 0.87986
dropout_lin = 0.087821
learning_rate = 0.00021999
weight_decay = 0.00029587
opt_step_size = 15
hidden_dim = 128
bidirectional = True
# bidirectional = False
lstm_layers = 2
lstm_output_dim= 129

hparams = Namespace(batch_size=batch_size,
                    dropout_lstm=dropout_lstm,
                    dropout_lin=dropout_lin,
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
                    )

model = MyNet(hparams) 
model.double()

train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True)
val   = DataLoader(myXYDataset(X["val"],   Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True)
test  = DataLoader(myXYDataset(X["test"],  Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True)

path_ckps = "./lightning_logs/test/"
# path_ckps = "./TEST_logs/lightning_logs/test/"

early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min', patience=5)
ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}", save_top_k=1, verbose=False, 
                      prefix="", monitor="loss", mode="min")

trainer = Trainer(gpus=0, min_epochs=1, max_epochs=2, deterministic=True, callbacks=[early_stop_callback, ckp])
# trainer = Trainer(gpus=0, min_epochs=1, max_epochs=2, deterministic=True, callbacks=early_stop_callback)


trainer.fit(model, train, val)
res = trainer.test(test_dataloaders=test)

# +
#to see the result of DataLoader

for i, batch in enumerate(train):
    print(i, batch)

# +
model.eval()
pred = model(torch.tensor(X["test"].values.astype(np.float)))
pred = pd.concat([test_pids, pd.Series(torch.round(torch.sigmoid(pred)).detach().view(-1))], axis=1)

pred.to_csv("/Users/fatemeh/Sleep Project/4_Sleep Boundary/sleep_boundary_project/data/files/part of raw data/predictions_nn.csv.gz", index=False)

# -


def eval_n_times(config, datafolder, n, gpus=1, patience=3):
    
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
    
    results = []
    for s in range(n):
        seed.seed_everything(s)

        path_ckps = "./lightning_logs/test/"
#         path_ckps = "./TEST_logs/lightning_logs/test/"
    
        if monitor == "mcc":
            early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='mcc', mode='max', patience=patience)
            ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False, prefix="",
                                  monitor="mcc", mode="max")
        else:
            early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min', patience=patience)
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

        trainer = Trainer(gpus=gpus, min_epochs=2, max_epochs=100, deterministic=True,
                          callbacks=[early_stop_callback, ckp])
        trainer.fit(model, train, val)
        res = trainer.test(test_dataloaders=test)
        results.append(res[0])
        
    return pd.DataFrame(results)


# +
def hyper_tuner(config, datafolder):
    
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

#     path_ckps = "./lightning_logs/test/"
    path_ckps = "./TEST_logs/lightning_logs/test/"
    
    if monitor == "mcc":
        early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='mcc', mode='max', patience=5)
        ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False, prefix="",
                              monitor="mcc", mode="max")
    else:
        early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min', patience=5)
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
    
    trainer = Trainer(gpus=0, min_epochs=2, max_epochs=100, deterministic=True,
                      callbacks=[early_stop_callback, ckp, tune_cb])
    trainer.fit(model, train, val)


# -

def run_tuning_procedure(datafolder, config, expname, ntrials, ncpus, ngpus):

    trainable = tune.with_parameters(hyper_tuner, datafolder=datafolder)

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
datafolder = "/home/palotti/github/sleep_boundary_project/data/processed_pycaret_5min/"
# datafolder = "/Users/fatemeh/Sleep Project/4_Sleep Boundary/sleep_boundary_project/data/files/part of raw data/processed_pycaret_5min/"

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

# ncpus=16
ncpus=4
# ngpus=3
ngpus=0
ntrials=1000
exp_name = "test"

run_tuning_procedure(datafolder, config_lstm, exp_name, ntrials=ntrials, ncpus=ncpus, ngpus=ngpus)



# +
datafolder = "/home/palotti/github/sleep_boundary_project/data/processed_pycaret_5min/"
# datafolder = "/Users/fatemeh/Sleep Project/4_Sleep Boundary/sleep_boundary_project/data/files/part of raw data/processed_pycaret_5min/"

best_parameters = {"batch_size": 1024, "bidirectional": True, "dropout_lin": 0.087821, 
                   "dropout_lstm": 0.87986, "hidden_dim": 128, "learning_rate": 0.00021999,
                   "lstm_layers": 2, "lstm_output_dim": 129, "monitor": "loss", 
                   "opt_step_size": 15, "weight_decay": 0.00029587}

results_MyNet_MP = eval_n_times(best_parameters, datafolder, 10, gpus=0, patience=3)

# -

results_MyNet_MP.mean(), results_MyNet_MP.std() 






