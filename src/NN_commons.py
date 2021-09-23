import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from pycaret.classification import *

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

from argparse import Namespace

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


def data_exists(datafolder, suffix):
    if not os.path.exists(datafolder):
        return False

    for f in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        if not os.path.exists("%s/%s_%s.csv.gz" % (datafolder, f, suffix)):
            return False

    return True


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


def pycater_setup(train_data, test_data,
                  gt_label="label_5min",
                  ignore_feat=["id", "fold", "linetime", "activity", "percentage_ground_truth"],
                  use_gpu=False):
    experiment = setup(data=train_data, test_data=test_data,
                       target=gt_label, session_id=123,
                       normalize=True, transformation=True,
                       fold_strategy="groupkfold", fold_groups="fold",
                       ignore_features=ignore_feat,
                       silent=True, use_gpu=use_gpu,
                       # normalize_method = 'zscore',
                       normalize_method='minmax',
                       # remove_outliers = True,
                       polynomial_features=True,
                       # fix_imbalance = True,
                       )
    return experiment


# +

# Pycaret initializer and auxiliar functions

def create_xy(train_path="train_data.csv.gz", test_path="test_data.csv.gz", use_gpu=False):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    test_pids = test_data[["pid", "gt_time"]]
    features_to_ignore = ["pid", "fold", "gt_time",
                          "percentage_ground_truth"] if "percentage_ground_truth" in train_data else ["pid", "fold",
                                                                                                      "gt_time"]

    experiment = pycater_setup(train_data, test_data,
                               gt_label="ground_truth",
                               ignore_feat=features_to_ignore,
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

    y_train['percentage_ground_truth'] = train_data['percentage_ground_truth']
    y_test['percentage_ground_truth'] = test_data['percentage_ground_truth']

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

        if break_point is None:
            break_point = input_size

        self.lstm = nn.LSTM(break_point, hidden_dim, num_layers=num_layers, dropout=dropout_lstm,
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

        if bidirectional:
            self.hn = torch.zeros(num_layers * 2, hidden_dim, hidden_dim).double()
            self.cn = torch.zeros(num_layers * 2, hidden_dim, hidden_dim).double()
        else:
            # Not sure why we need to multiply num_layers * hidden_dim here and not on the if above
            self.hn = torch.zeros(num_layers, num_layers * hidden_dim, hidden_dim).double()
            self.cn = torch.zeros(num_layers, num_layers * hidden_dim, hidden_dim).double()

        nn.init.xavier_normal_(self.hn)
        nn.init.xavier_normal_(self.cn)

        self.last_lin = nn.Sequential(nn.Linear(last_d, output_dim), nn.ReLU(inplace=True))
        self.break_point = break_point

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1] // self.break_point, -1)

        x, _ = self.lstm(x)

        # x, (self.hn, self.cn) = self.lstm(x, (self.hn, self.cn))
        x = x.reshape(x.shape[0], -1)

        for lay in self.linlayers:
            x = lay(x)

        x = self.last_lin(x)
        return x


# +
def eval_n_times(MyNet, config, datafolder, featset, n, gpus=1, patience=5, min_epochs=2, max_epochs=20):
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

    X, Y, test_pids = load_data(datafolder, featset)

    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True,
                       num_workers=8)
    val = DataLoader(myXYDataset(X["val"], Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True,
                     num_workers=8)
    test = DataLoader(myXYDataset(X["test"], Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True,
                      num_workers=8)

    results = []
    for s in range(n):
        seed.seed_everything(s)

        #         path_ckps = "./lightning_logs/test/"
        path_ckps = "./TEST_logs/lightning_logs/test/"

        early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min',
                                            patience=patience)
        ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False,
                              monitor="loss", mode="min")

        hparams = Namespace(batch_size=batch_size,
                            input_dim=X["train"].shape[1],
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

        trainer = Trainer(gpus=gpus, min_epochs=min_epochs, max_epochs=max_epochs, deterministic=True,
                          callbacks=[early_stop_callback, ckp])
        trainer.fit(model, train, val)
        res = trainer.test(test_dataloaders=test)
        results.append(res[0])

    return pd.DataFrame(results)


# +
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
    weight_decay = config["weight_decay"]

    labels = config["labels"]
    classification_tasks = config["classification_tasks"]
    regression_tasks = config["regression_tasks"]
    weights = config["weights"]
    loss_fnct = config["loss_fnct"]

    X, Y, test_pids = load_data(datafolder, featset)

    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True,
                       num_workers=8)
    val = DataLoader(myXYDataset(X["val"], Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True,
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
                        #
                        labels=labels,
                        loss_fnct=loss_fnct,
                        regression_tasks=regression_tasks,
                        classification_tasks=classification_tasks,
                        weights=weights,
                        )

    model = MyNet(hparams)
    model.double()

    tune_metrics = {"loss": "loss", "mcc_main_y": "mcc_main_y", "acc_main_y": "acc_main_y",
                    "prec_main_y": "prec_main_y",
                    "rec_main_y": "rec_main_y", "f1_main_y": "f1_main_y",
                    "MAE_y": "MAE_y", "MSE_y": "MSE_y", "r2_y": "r2_y"}

    tune_cb = TuneReportCallback(tune_metrics, on="validation_end")

    trainer = Trainer(gpus=gpu_per_node, min_epochs=min_epochs, max_epochs=max_epochs, deterministic=True,
                      callbacks=[early_stop_callback, ckp, tune_cb])
    trainer.fit(model, train, val)


# -

def run_tuning_procedure(MyNet, datafolder, featset, config, expname, ntrials, ncpus, ngpus,
                         min_epochs=1,
                         max_epochs=20):
    trainable = tune.with_parameters(hyper_tuner,
                                     featset=featset,
                                     MyNet=MyNet,
                                     datafolder=datafolder,
                                     min_epochs=min_epochs,
                                     max_epochs=max_epochs,
                                     gpu_per_node=0)

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
    print(analysis.results_df.sort_values(by="loss", ascending=False).head(5))
