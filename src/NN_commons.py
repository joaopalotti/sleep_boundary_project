from argparse import Namespace

import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import seed


def calculate_classification_metrics(labels, predictions):
    return metrics.accuracy_score(labels, predictions), \
           metrics.precision_score(labels, predictions), \
           metrics.recall_score(labels, predictions), \
           metrics.f1_score(labels, predictions, average='weighted'), \
           metrics.matthews_corrcoef(labels, predictions)


def calculate_regression_metrics(labels, predictions):
    return metrics.mean_absolute_error(labels, predictions), \
           metrics.mean_squared_error(labels, predictions), \
           metrics.r2_score(labels, predictions)


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


class finalTestDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X.iloc[idx].values.astype(np.double)
        return x



def save_data(output_folder, X, Y, test_pids, featset):
    X["train"].to_csv("%s/X_train_%s.csv.gz" % (output_folder, featset), index=False)
    X["val"].to_csv("%s/X_val_%s.csv.gz" % (output_folder, featset), index=False)
    X["test"].to_csv("%s/X_test_%s.csv.gz" % (output_folder, featset), index=False)

    Y["train"].to_csv("%s/y_train_%s.csv.gz" % (output_folder, featset), index=False)
    Y["val"].to_csv("%s/y_val_%s.csv.gz" % (output_folder, featset), index=False)
    Y["test"].to_csv("%s/y_test_%s.csv.gz" % (output_folder, featset), index=False)

    test_pids.to_csv("%s/test_pids_%s.csv.gz" % (output_folder, featset))


def load_data(datafolder, featset):
    X, Y = {}, {}
    X["train"] = pd.read_csv("%s/X_train_%s.csv.gz" % (datafolder, featset))
    X["val"] = pd.read_csv("%s/X_val_%s.csv.gz" % (datafolder, featset))
    X["test"] = pd.read_csv("%s/X_test_%s.csv.gz" % (datafolder, featset))

    Y["train"] = pd.read_csv("%s/y_train_%s.csv.gz" % (datafolder, featset))
    Y["val"] = pd.read_csv("%s/y_val_%s.csv.gz" % (datafolder, featset))
    Y["test"] = pd.read_csv("%s/y_test_%s.csv.gz" % (datafolder, featset))

    test_pids = pd.read_csv("%s/test_pids_%s.csv.gz" % (datafolder, featset))

    return X, Y, test_pids



def get_number_internal_layers(n, output_size):
    """
    E.g.:
        get_number_internal_layers(20, 3) --> [16, 8, 4]
        get_number_internal_layers(192, 16) # --> [128, 64, 32]
    """
    i = 1;
    d = 2;
    s = []
    while (n - 1) / d > 1:
        s.append(d)
        i += 1
        d = 2 ** i

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

        self.hidden_dim = hidden_dim

        if break_point is None:
            break_point = input_size

        self.lstm = nn.LSTM(break_point, self.hidden_dim,
                            num_layers=num_layers, dropout=dropout_lstm,
                            batch_first=True, bidirectional=bidirectional)
        self.linlayers = nn.ModuleList()
        self.drop = nn.Dropout(dropout_lin)

        if bidirectional:
            hidden_dim *= 2

        last_d = hidden_dim * (input_size // break_point)
        for lay_size in get_number_internal_layers(last_d, output_dim):
            print("Last: %d, Next: %d" % (last_d, lay_size))
            self.linlayers.append(nn.Sequential(nn.Linear(last_d, lay_size),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(dropout_lin))
                                  )

            last_d = lay_size

        print("Very Last: %d, Out: %d" % (last_d, output_dim))
        print("#Lin layers: ", len(self.linlayers))

        self.last_lin = nn.Sequential(nn.Linear(last_d, output_dim), nn.ReLU(inplace=True))
        self.break_point = break_point

    def forward(self, x, hs=None):
        batch_size = x.size(0)

        x = x.view(batch_size, x.shape[1] // self.break_point, -1)
        x, hs = self.lstm(x, hs)

        x = x.view(batch_size, -1)

        for lay in self.linlayers:
            x = lay(x)

        x = self.last_lin(x)
        return x, hs


# -

def eval_n_times(MyNet, config, datafolder, featset, n, gpus=1, patience=5,
                 min_epochs=2, max_epochs=20, save_predictions=False):
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

    labels = config["labels"]
    classification_tasks = config["classification_tasks"]
    regression_tasks = config["regression_tasks"]
    main_weight = config["main_weight"]
    loss_fnct = config["loss_fnct"]

    cnn_layers = config["cnn_layers"]
    cnn_kernel_size = config["cnn_kernel_size"]

    X, Y, test_pids = load_data(datafolder, featset)

    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=int(batch_size), shuffle=True, drop_last=False, num_workers=8)
    val = DataLoader(myXYDataset(X["val"], Y["val"]), batch_size=int(batch_size), shuffle=False, drop_last=False, num_workers=8)
    test = DataLoader(myXYDataset(X["test"], Y["test"]), batch_size=int(batch_size), shuffle=False, drop_last=False, num_workers=8)


    results = []
    for s in range(n):
        seed.seed_everything(s)

        #         path_ckps = "./lightning_logs/test/"
        path_ckps = "./TEST_logs/lightning_logs/test/"

        early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min',
                                            patience=patience)
        ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False,
                              monitor="loss", mode="min")

        hparams = Namespace(batch_size=int(batch_size),
                            input_dim=X["train"].shape[1],
                            # Optmizer configs
                            opt_learning_rate=float(learning_rate),
                            opt_step_size=int(opt_step_size),
                            opt_gamma=0.5,
                            cnn_layers=int(cnn_layers),
                            cnn_kernel_size=int(cnn_kernel_size),
                            # LSTM configs
                            hidden_dim=int(hidden_dim),
                            bidirectional=bool(bidirectional),
                            lstm_layers=int(lstm_layers),
                            lstm_output_dim=int(lstm_output_dim),
                            dropout_lstm=float(dropout_lstm),
                            dropout_lin=float(dropout_lin),
                            #
                            labels=list(labels),
                            loss_fnct=list(loss_fnct),
                            regression_tasks=list(regression_tasks),
                            classification_tasks=list(classification_tasks),
                            main_weight=float(main_weight),
                            )

        model = MyNet(hparams)
        model.double()

        trainer = Trainer(gpus=gpus, min_epochs=min_epochs, max_epochs=max_epochs, deterministic=True,
                          callbacks=[early_stop_callback, ckp])
        trainer.fit(model, train, val)
        res = trainer.test(test_dataloaders=test)
        results.append(res[0])

        if save_predictions is not None:
            finaltest = DataLoader(finalTestDataset(X["test"]), batch_size=1, shuffle=False, drop_last=False, num_workers=8)
            predictions = trainer.predict(model, dataloaders=finaltest, return_predictions=True)
            ks = predictions[0].keys()

            r = {}
            for k in ks:
                r[k] = pd.concat([pd.Series(e[k].view(-1).numpy()) for e in predictions]).reset_index(drop=True)
            predictions = pd.DataFrame(r)
            predictions = pd.concat([test_pids, predictions], axis=1)
            predictions.to_csv(save_predictions, index=False)

    return pd.DataFrame(results)


def hyper_tuner(config, MyNet, datafolder, featset, min_epochs, max_epochs, gpu_per_node=0):
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

    labels = config["labels"]
    classification_tasks = config["classification_tasks"]
    regression_tasks = config["regression_tasks"]
    main_weight = config["main_weight"]
    loss_fnct = config["loss_fnct"]

    cnn_layers = config["cnn_layers"]
    cnn_kernel_size = config["cnn_kernel_size"]

    X, Y, test_pids = load_data(datafolder, featset)

    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=False,
                       num_workers=8)
    val = DataLoader(myXYDataset(X["val"], Y["val"]), batch_size=batch_size, shuffle=False, drop_last=False,
                     num_workers=8)

    seed.seed_everything(42)

    path_ckps = "./TEST_logs/lightning_logs/test/"

    early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min', patience=5)
    ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False,
                          monitor="loss", mode="min")

    hparams = Namespace(batch_size=batch_size,
                        input_dim=X["train"].shape[1],
                        #
                        # Optmizer configs
                        #
                        opt_learning_rate=learning_rate,
                        opt_step_size=opt_step_size,
                        opt_gamma=0.5,
                        cnn_layers=cnn_layers,
                        cnn_kernel_size=cnn_kernel_size,
                        # LSTM configs
                        hidden_dim=hidden_dim,
                        bidirectional=bidirectional,
                        lstm_layers=lstm_layers,
                        lstm_output_dim=lstm_output_dim,
                        dropout_lstm=dropout_lstm,
                        dropout_lin=dropout_lin,
                        #
                        labels=labels,
                        loss_fnct=loss_fnct,
                        regression_tasks=regression_tasks,
                        classification_tasks=classification_tasks,
                        main_weight=main_weight,
                        )

    model = MyNet(hparams)
    model.double()

    tune_metrics = {"loss": "loss"}

    for task in classification_tasks:
        tune_metrics.update(
            {"mcc_%s" % task: "mcc_%s" % task, "acc_%s" % task: "acc_%s" % task, "prec_%s" % task: "prec_%s" % task,
             "rec_%s" % task: "rec_%s" % task, "f1_%s" % task: "f1_%s" % task})

    for task in regression_tasks:
        tune_metrics.update(
            {"MAE_%s" % task: "MAE_%s" % task, "MSE_%s" % task: "MSE_%s" % task, "r2_%s" % task: "r2_%s" % task})

    tune_cb = TuneReportCallback(tune_metrics, on="validation_end")

    trainer = Trainer(gpus=gpu_per_node, min_epochs=min_epochs, max_epochs=max_epochs, deterministic=True,
                      callbacks=[early_stop_callback, ckp, tune_cb])
    trainer.fit(model, train, val)


def run_tuning_procedure(MyNet, datafolder, featset, config, expname, ntrials, ncpus, ngpus,
                         min_epochs=1, max_epochs=20):
    trainable = tune.with_parameters(hyper_tuner,
                                     featset=featset,
                                     MyNet=MyNet,
                                     datafolder=datafolder,
                                     min_epochs=min_epochs,
                                     max_epochs=max_epochs,
                                     gpu_per_node=0)

    analysis = tune.run(trainable,
                        resources_per_trial={"cpu": ncpus, "gpu": ngpus},
                        # metric="loss", mode="min",
                        metric="mcc_main_y", mode="max",
                        config=config,
                        num_samples=ntrials,
                        name=expname)

    print("Best Parameters:", analysis.best_config)

    analysis.best_result_df.to_csv("best_parameters_exp%s_trials%d.csv" % (expname, ntrials))
    analysis.results_df.to_csv("all_results_exp%s_trials%d.csv" % (expname, ntrials))
    print("Best 5 results")
    print(analysis.results_df.sort_values(by="mcc_main_y", ascending=False).head(5))
    return analysis.best_result_df

