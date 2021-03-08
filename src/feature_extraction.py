# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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
import tempfile
import pandas as pd
pd.set_option('display.max_rows', 20)

import numpy as np
import os
from tqdm import tqdm
from glob import glob
from datetime import timedelta
from tsfresh import select_features, extract_features
from tsfresh.utilities.dataframe_functions import impute

from sklearn.model_selection import KFold


# +
def read_file(filename):
    df = pd.read_csv(filename)
    df["sleep"] = df["stages"] > 0
    df["linetime"] = pd.to_datetime(df["linetime"])
    return df

df = read_file("../data/processed/mesa/0001.csv.gz")


# -

def generate_slide_wins(df, winsize=11):
    
    seq_id = 0
    transformed_df = []
    list_of_indexes=[]
    labels = []
    df.index.to_series().rolling(winsize, center=True).apply((lambda x: list_of_indexes.append(x.tolist()) or 0), raw=False)
    for idx in list_of_indexes:
        labels.append(df.iloc[idx].iloc[winsize//2]["ground_truth"])
        tmp_df = df.iloc[idx].copy()
        tmp_df["seq_id"] = seq_id
        seq_id += 1
        transformed_df.append(tmp_df)

    return pd.concat(transformed_df), pd.Series(labels)


# +
# This cell will take many many many many hours to run....we can think of better ways to process it
input_files = glob("../data/processed/mesa/*.csv.gz")
all_ys = []
all_features = []
all_ids = []

for file in input_files:
    df = read_file(file)
    transformed_df, labels = generate_slide_wins(df, 21)

    extracted_features = extract_features(transformed_df[["activity", "mean_hr", "linetime", "seq_id"]], 
                                     column_id="seq_id", column_sort="linetime")
    
    impute(extracted_features)
    features_filtered = select_features(extracted_features, labels)
    
    ids = pd.Series(labels.shape[0]*[df["mesaid"].unique()[0]])
    all_features.append(features_filtered)
    all_ys.append(labels)
    all_ids.append(ids)


# +
# This process took long time, but I am saving the final dataframes to files here...
# Just need to load those into memory...

# all_features.to_csv("all_features_win21.csv.gz", index=False)
# all_ys.to_csv("all_ys_win21.csv.gz", index=False)
# all_ids.to_csv("all_ids_win21.csv.gz", index=False)

# +
# TODO: use sklearn pipelines to perform a 5-CV evaluation and predictions of the labels
#all_features = pd.read_csv("all_features_win21.csv.gz")
#all_ys = pd.read_csv("all_ys_win21.csv.gz")
#all_ids = pd.read_csv("all_ids_win21.csv.gz")

# +
def map_id_fold(all_ids, n): 
    pids = all_ids["id"].unique().ravel()
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    mapping = []
    for i, (_, test) in enumerate(kf.split(pids)):
        for pid_index in test:
            mapping.append({'fold': i, 'id': pids[pid_index]})

    return pd.DataFrame(mapping)

df_pid_fold = map_id_fold(all_ids, 2)
df_pid_fold = pd.merge(df_pid_fold, all_ids)

all_data = pd.concat([df_pid_fold.reset_index(drop=True), all_ys.reset_index(drop=True), all_features.reset_index(drop=True)], axis=1)


# +
#all_data = all_data[:300000]

# +
from pycaret.classification import *

experiment = setup(data=all_data, #test_data=test_data,
                   target="sleep", 
                   session_id=123,
                   normalize=True,
                   transformation=True,
                   fold_strategy="groupkfold",
                   fold_groups="fold",
                   ignore_features=["id"],
                   silent=True,
                   use_gpu=False
                   )

#lr = LinearRegression()
#lr.fit(all_features[:1000].fillna(0.0), all_ys[:1000])

# -

best = compare_models()

create_model("lightgbm")
