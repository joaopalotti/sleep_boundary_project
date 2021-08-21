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

from sklearn.model_selection import KFold

from argparse import Namespace
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
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

def extract_features(path_to_raw_features="../data/files/", 
                    signals = ["activity", "mean_hr"]):
    
    train_data = pd.read_csv(os.path.join(path_to_raw_features, "train_raw_data.csv.gz"))
    test_data = pd.read_csv(os.path.join(path_to_raw_features, "test_raw_data.csv.gz"))
    
    # Extract val data from train_data
    val_data = train_data[train_data["fold"] == 9]
    train_data = train_data[train_data["fold"] != 9]

    # Get Y
    y_train = train_data[["ground_truth"]]
    y_val = val_data[["ground_truth"]]
    y_test = test_data[["ground_truth"]]
    
    # List Fetures    
    features = [c for s in signals for c in train_data.keys() if c.startswith(s)]
    
    # Normalize features
    scaler = preprocessing.MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train_data[features].values))
    X_val = pd.DataFrame(scaler.fit_transform(val_data[features].values))
    X_test = pd.DataFrame(scaler.transform(test_data[features].values))
    
    test_pids = test_data[["pid", "gt_time"]]
    
    X = {"train": X_train, "val": X_val, "test": X_test}
    Y = {"train": y_train, "val": y_val, "test": y_test}
    
    return X, Y, test_pids



# +
def data_exists(datafolder, suffix="raw"):
    if not os.path.exists(datafolder):
        return False
    
    for f in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]: 
        if not os.path.exists("%s/%s_%s.csv.gz" % (datafolder, f, suffix)):
            return False
        
    return True

def save_data(output_folder, X, Y, test_pids):
    X["train"].to_csv("%s/X_train_raw.csv.gz" % (output_folder), index=False)
    X["val"].to_csv("%s/X_val_raw.csv.gz" % (output_folder), index=False)
    X["test"].to_csv("%s/X_test_raw.csv.gz" % (output_folder), index=False)

    Y["train"].to_csv("%s/y_train_raw.csv.gz" % (output_folder), index=False)
    Y["val"].to_csv("%s/y_val_raw.csv.gz" % (output_folder), index=False)
    Y["test"].to_csv("%s/y_test_raw.csv.gz" % (output_folder), index=False)

    test_pids.to_csv("%s/test_pids.csv.gz" % (output_folder))
    
def load_data(datafolder):
    X, Y = {}, {}
    X["train"] = pd.read_csv("%s/X_train_raw.csv.gz" % (datafolder))
    X["val"] = pd.read_csv("%s/X_val_raw.csv.gz" % (datafolder))
    X["test"] = pd.read_csv("%s/X_test_raw.csv.gz" % (datafolder))
    
    Y["train"] = pd.read_csv("%s/y_train_raw.csv.gz" % (datafolder))
    Y["val"] = pd.read_csv("%s/y_val_raw.csv.gz" % (datafolder))
    Y["test"] = pd.read_csv("%s/y_test_raw.csv.gz" % (datafolder))
    
    test_pids = pd.read_csv("%s/test_pids.csv.gz" % (datafolder))
    
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

def calculate_regression_metrics(labels, predictions):

    return metrics.mean_absolute_error(labels, predictions),\
            metrics.mean_squared_error(labels, predictions),\
            metrics.r2_score(labels, predictions)


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

        x = x.view(x.shape[0], x.shape[1]//self.break_point, -1)
        x, hidden = self.lstm(x)
        x = x.reshape(x.shape[0], -1)
        
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
        self.net = LSTMLayer(input_size=42, break_point=21,
                             dropout_lstm=self.dropout_lstm,
                             dropout_lin=self.dropout_lin,
                             hidden_dim=self.hidden_dim,
                             bidirectional=self.bidirectional,
                             num_layers=self.lstm_layers,
                             output_dim=self.lstm_output_dim,
                            )
        
        self.drop = nn.Dropout(self.dropout_lin)
        self.head = nn.ModuleDict()
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min',
                                                         patience=self.opt_step_size,
                                                         factor=self.opt_gamma, # new_lr = lr * factor (default = 0.1)
                                                         verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}
    
    
    def range_normalization(tensor):
        tensor = tensor.view(tensor.size(0))
        tensor -= tensor.min()
        tensor /= tensor.max()
        tensor = tensor.view(-1,1)
    
        return tensor
    
    def calculate_losses(self, y, predictions):
        loss_fnct = nn.BCEWithLogitsLoss()
        label = {}
        label['main_y'] = y[0: , 0:-1]
        label['percentage_y'] = y[0: , 1:]
        
        weights = {}
        weights['main_y'] = 0.75
        weights['percentage_y'] = 0.25
        
        keys = ['main_y', 'percentage_y']
        loss = {}
        final_loss = 0
        for key in keys:
            if(key != 'main_y'):
                predictions[key]  = range_normalization(predictions[key])
                label[key]  = range_normalization(label[key])
                
            loss[key] = loss_fnct(predictions[key], label[key])
            final_loss +=  weights[key] * loss[key] 

        return final_loss

    def process_step(self, batch):
        x, y = batch
 
        predictions = {}
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
        y['main_y'] = torch.stack([row["y"][0: , 0:-1] for row in outputs]).view(-1).cpu()
        y['percentage_y'] = torch.stack([row["y"][0: , 1:] for row in outputs]).view(-1).cpu()


        pred = {}     
        pred['main_y'] = torch.stack([row["preds"]['main_y'] for row in outputs]).view(-1)
        pred['percentage_y'] = torch.stack([row["preds"]['percentage_y'] for row in outputs]).view(-1)


        pred['main_y'] = torch.round(torch.sigmoid(pred['main_y']))
        pred['main_y'] = pred['main_y'] .cpu()
        pred['percentage_y'] = pred['percentage_y'].cpu()
        
        key_list = ['main_y', 'percentage_y']


        acc_main_y, prec_main_y, rec_main_y, f1_main_y, mcc_main_y = calculate_classification_metrics(y['main_y'], pred['main_y'])
        
        self.log("acc_main_y", acc_main_y)
        self.log("prec_main_y", prec_main_y)
        self.log("rec_main_y", rec_main_y)
        self.log("f1_main_y", f1_main_y)
        self.log("mcc_main_y", mcc_main_y)
        
        print("(Val_main_y) Epoch: %d, Acc: %.3f, P: %.3f, R: %.3f, F1: %.3f, MCC: %.3f" % (self.current_epoch,
                                                                                     acc_main_y, prec_main_y, rec_main_y, f1_main_y, mcc_main_y))

        
        
        MAE_y, MSE_y, r2_y  = calculate_regression_metrics(y['percentage_y'], pred['percentage_y'])
        
        self.log("MAE_y", MAE_y)
        self.log("MSE_y", MSE_y)
        self.log("MSE_y", r2_y)
        
        print("(Val_percentage_y) Epoch: %d, MAE_y: %.3f, MSE_y: %.3f, r2: %.3f" % (self.current_epoch,
                                                                                     MAE_y, MSE_y, r2_y))
             
    def test_epoch_end(self, outputs):
        test_loss = torch.stack([row['loss'] for row in outputs]).mean()
        print("(Test) Total Loss: %.4f" % test_loss)

        y = {}
        y['main_y'] = torch.stack([row["y"][0: , 0:-1] for row in outputs]).view(-1).cpu()
        y['percentage_y'] = torch.stack([row["y"][0: , 1:] for row in outputs]).view(-1).cpu()
        

        pred = {}     
        pred['main_y'] = torch.stack([row["preds"]['main_y'] for row in outputs]).view(-1)
        pred['percentage_y'] = torch.stack([row["preds"]['percentage_y'] for row in outputs]).view(-1)
        pred['main_y'] = torch.round(torch.sigmoid(pred['main_y']))
        pred['main_y'] = pred['main_y'] .cpu()
        pred['percentage_y'] = pred['percentage_y'].cpu()
        
            
        acc_main_y, prec_main_y, rec_main_y, f1_main_y, mcc_main_y = calculate_classification_metrics(y['main_y'], pred['main_y'])
        
        self.log("acc_main_y", acc_main_y)
        self.log("prec_main_y", prec_main_y)
        self.log("rec_main_y", rec_main_y)
        self.log("f1_main_y", f1_main_y)
        self.log("mcc_main_y", mcc_main_y)
        print("TEST: Acc: %.3f, P: %.3f, R: %.3f, F1: %.3f, MCC: %.3f" % (acc_main_y, prec_main_y, rec_main_y, f1_main_y, mcc_main_y))
        
        
        MAE_y, MSE_y, r2_y  = calculate_regression_metrics(y['percentage_y'], pred['percentage_y'])
        
        self.log("MAE_y", MAE_y)
        self.log("MSE_y", MSE_y)
        self.log("MSE_y", r2_y)
        
        print("(TEST: MAE_y: %.3f, MSE_y: %.3f, r2: %.3f" % (MAE_y, MSE_y, r2_y))

# ### The next two cells are able to run the network one single time. 
# It is useful for us to debug the network before running the param tuning

# +
datafolder = "../data/processed/raw_5min/."

if data_exists(datafolder):
    X, Y, test_pids = load_data(datafolder)
else:
    X, Y, test_pids = extract_features()
    save_data(datafolder, X, Y, test_pids)

# +
batch_size = 1024
dropout_lstm = 0.87986
dropout_lin = 0.087821
learning_rate = 0.00021999
weight_decay = 0.00029587
opt_step_size = 15
hidden_dim = 128
bidirectional = True
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

train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
val   = DataLoader(myXYDataset(X["val"],   Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
test  = DataLoader(myXYDataset(X["test"],  Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)

path_ckps = "./lightning_logs/test/"

early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min', patience=5)
ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}", save_top_k=1, verbose=False, 
                      prefix="", monitor="loss", mode="min")

trainer = Trainer(gpus=0, min_epochs=1, max_epochs=10, deterministic=True, callbacks=[early_stop_callback, ckp])
trainer.fit(model, train, val)
res = trainer.test(test_dataloaders=test)

# +
model.eval()
pred = model(torch.tensor(X["test"].values.astype(np.float)))
pred = pd.concat([test_pids, pd.Series(torch.round(torch.sigmoid(pred)).view(-1))], axis=1)

pred.to_csv("predictions_raw_nn.csv.gz", index=False)


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
    
    X, Y, test_pids = load_data(datafolder)
    
    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val   = DataLoader(myXYDataset(X["val"],   Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    test  = DataLoader(myXYDataset(X["test"],  Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    
    results = []
    for s in range(n):
        seed.seed_everything(s)

        path_ckps = "./lightning_logs/test/"

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
    
    X, Y, test_pids = load_data(datafolder)
    
    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val   = DataLoader(myXYDataset(X["val"],   Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    test  = DataLoader(myXYDataset(X["test"],  Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    
    seed.seed_everything(42)

    path_ckps = "./lightning_logs/test/"

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
#datafolder = "/home/palotti/github/sleep_boundary_project/data/processed_pycaret_5min/"
datafolder = "/export/sc2/jpalotti/github/sleep_boundary_project/data/processed/raw_5min/"

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

ncpus=16
ngpus=3
ntrials=1000
exp_name = "test"

run_tuning_procedure(datafolder, config_lstm, exp_name, ntrials=ntrials, ncpus=ncpus, ngpus=ngpus)


# +
datafolder = "/home/palotti/github/sleep_boundary_project/data/processed_pycaret_5min/"
best_parameters = {"batch_size": 1024, "bidirectional": True, "dropout_lin": 0.087821, 
                   "dropout_lstm": 0.87986, "hidden_dim": 128, "learning_rate": 0.00021999,
                   "lstm_layers": 2, "lstm_output_dim": 129, "monitor": "loss", 
                   "opt_step_size": 15, "weight_decay": 0.00029587}

results_MyNet_MP = eval_n_times(best_parameters, datafolder, 10, gpus=0, patience=3)
# -

results_MyNet_MP.mean(), results_MyNet_MP.std() 


