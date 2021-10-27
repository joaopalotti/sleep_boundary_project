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
import os, sys
from tqdm import tqdm
from glob import glob

from pycaret.classification import *
from sklearn.model_selection import KFold


# -
def get_env_var(varname, default):
    return int(os.environ.get(varname)) if os.environ.get(varname) is not None else default

def chunks(l, n):
    n = len(l) // n
    return [l[i:i+n] for i in range(0, len(l), max(1, n))]


def pycater_setup(train_data, test_data,
                  gt_label="label_5min",
                  ignore_feat=["id", "fold", "linetime", "activity", "gt_time"],
                  use_gpu=False, n_jobs=-1):
    if "percentage_ground_truth" in train_data:
        ignore_feat.append("percentage_ground_truth")

    experiment = setup(data=train_data, test_data=test_data,
                       target=gt_label, session_id=123,
                       normalize=True,
                       transformation=False,
                       fold_strategy="groupkfold",
                       fold_groups="fold",
                       ignore_features=ignore_feat,
                       silent=True,
                       use_gpu=use_gpu,
                       normalize_method='robust',
                       remove_outliers = False,
                       polynomial_features=False,
                       fix_imbalance = False,
                       n_jobs=n_jobs
                       )
    return experiment


if __name__ == "__main__":

    NCPUS = get_env_var('SLURM_CPUS_PER_TASK', 12)
    NGPUS = 1
    NTRIALS = 20
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    n_iter = 20

    combinations = []
    for model in ["lr", "rf", "et", "lda", "lightgbm"]:
        for win in ["10min_centered", "20min_centered", "40min_centered", "10min_notcentered",
                    "20min_notcentered", "40min_notcentered"]:
            for featset in ["raw", "tsfresh"]:
                combinations.append((model, win, featset))

    print("Total combinations:", len(combinations))
    print("All combinations:", combinations)
    selected_combinations = chunks(combinations, SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
    print("Processing: ", selected_combinations)

    for comb in selected_combinations:
        m, win, featset = comb

        #exp = "%s" % sys.argv[1] #10min_centered, 10min_notcentered...
        #featset = sys.argv[2] # "tsfresh", "raw"
        #n_jobs = int(sys.argv[3])

        datapath = "/export/sc2/jpalotti/github/sleep_boundary_project/data/processed/train_test_splits/%s/" % (win)
        print("Running with %s %s %s" % (win, featset, datapath))

        train_data = pd.read_csv(os.path.join(datapath, "train_%s_data.csv.gz" % featset))
        test_data = pd.read_csv(os.path.join(datapath, "test_%s_data.csv.gz" % featset))

        experiment = pycater_setup(train_data, test_data, gt_label="ground_truth", ignore_feat=["pid", "fold"], use_gpu=NGPUS>0, n_jobs=NCPUS)

        experiment_filename = "sleep_ml_%s_%s_%s" % (m, win, featset)
        print("Creating a %s model." % m)
        model = create_model(m)
        model = tune_model(model, n_iter=n_iter, choose_better=True)

        print("Creating final model to save results to disk............")
        model = create_model(model)

        dfresult = pull()
        dfresult["model"] = m
        dfresult["window"] = win
        dfresult["featset"] = featset
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
        dfresult["window"] = win
        dfresult["featset"] = featset
        dfresult.to_csv("%s_test.csv.gz" % experiment_filename, index=False)
        predictions[["Label", "Score"]].to_csv("%s_predictions.csv.gz" % experiment_filename)

        print("Saved TEST results to: %s_predictions.csv.gz" % experiment_filename)
