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


#all_data = pd.read_csv("all_data.csv.gz")
train_data = pd.read_csv("train_data.csv.gz")
test_data = pd.read_csv("test_data.csv.gz")

experiment = pycater_setup(train_data, test_data, 
                           gt_label = "ground_truth", ignore_feat = ["pid", "fold"])

# +
for m in tqdm(["lr", "rf", "et", "lda", "catboost", "lightgbm"]):

    experiment_filename = "sleep_ml_%s" % (m)
    print("Creating a %s model." % (m))
    model = create_model(m)
    model = tune_model(lr_model, n_iter=20, choose_better=True)

    print("Creating final model to save results to disk............")
    model = create_model(model)

    dfresult = pull()
    dfresult["model"] = m
    dfresult["X_shape"] = get_config("X").shape[0]
    dfresult["y_train_shape"] = get_config("y_train").shape[0]
    dfresult["y_test_shape"] = get_config("y_test").shape[0]
    dfresult.to_csv("%s.csv.gz" % experiment_filename, index=False)
    print("Saved results to: %s.csv.gz" % experiment_filename)

    predictions = predict_model(model)

    dfresult = pull()
    dfresult["model"] = m
    dfresult["test"] = True
    dfresult["X_shape"] = get_config("X").shape[0]
    dfresult["y_train_shape"] = get_config("y_train").shape[0]
    dfresult["y_test_shape"] = get_config("y_test").shape[0]
    dfresult.to_csv("%s_test.csv.gz" % experiment_filename, index=False)
    predictions[["Label", "Score"]].to_csv("%s_predictions.csv.gz" % experiment_filename)

    print("Saved TEST results to: %s_predictions.csv.gz" % experiment_filename)


