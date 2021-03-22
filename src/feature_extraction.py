# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
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
from tsfresh.feature_extraction import MinimalFCParameters

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
# to prevent from having many features, we are using MinimalFCParameters 

input_files = glob("../data/processed/mesa/*.csv.gz")
all_ys = []
all_features = []
all_ids = []

for file in input_files:
    df = read_file(file)
    transformed_df, labels = generate_slide_wins(df, 21)
    
    extracted_features = extract_features(transformed_df[["activity", "mean_hr", "linetime", "seq_id"]], column_id="seq_id",
                        column_sort="linetime", default_fc_parameters=MinimalFCParameters())

    
    impute(extracted_features)
    
    ids = pd.Series(labels.shape[0]*[df["mesaid"].unique()[0]])
    all_features.append(extracted_features)
    all_ys.append(labels)
    all_ids.append(ids)


# +
# concat all features, ids, and labels

# all_ids = pd.concat(all_ids)
# all_features = pd.concat(all_features)
# all_ys = pd.concat(all_ys)

# +
# write into a csv

# all_features.to_csv("all_features_win21.csv.gz", index=False)
# all_ys.to_csv("all_ys_win21.csv.gz", index=False)
# all_ids.to_csv("all_ids_win21.csv.gz", index=False)

# +
# # read them

# all_features = pd.read_csv("../data/all_features/all_features_win21.csv.gz")
# all_ids = pd.read_csv("../data/all_features/all_ids_win21.csv.gz")
# all_ys = pd.read_csv("../data/all_features/all_ys_win21.csv.gz")

# # change column names
# all_ids.columns = ['id']
# all_ys.columns = ['label']


# -

#read all_data
all_data = pd.read_csv("../data/all_data_win21.csv.gz")
all_data

# TODO: use sklearn pipelines to perform a 5-CV evaluation and predictions of the labels


# +
def map_id_fold(all_ids, n): 
    pids = all_ids["id"].unique().ravel()
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    mapping = []
    for i, (_, test) in enumerate(kf.split(pids)):
        for pid_index in test:
            mapping.append({'fold': i, 'id': pids[pid_index]})

    return pd.DataFrame(mapping)

df_pid_fold = map_id_fold(all_ids, 10)
df_pid_fold = pd.merge(df_pid_fold, all_ids)

all_data = pd.concat([df_pid_fold.reset_index(drop=True), all_ys.reset_index(drop=True), all_features.reset_index(drop=True)], axis=1)


# +
# all_data
# all_data.to_csv("../data/all_data_win21.csv.gz", index=False)

# +
#all_data = all_data[:300000]

# +
from pycaret.classification import *

experiment = setup(data=all_data, #test_data=test_data,
                   target="label",
                   train_size = 0.98, # when the data is bigger than 1M it counts as Big Data
                   session_id=123,
                   normalize=True,
                   transformation=True,
                   fold_strategy="groupkfold",
                   fold_groups="fold",
                   ignore_features=["id"],
                   silent=True,
                   use_gpu=False,
#                    normalize_method = 'zscore',
                   normalize_method = 'minmax',       
#                    remove_outliers = True,
                   polynomial_features = True,
#                    fix_imbalance = True,
                   )

#lr = LinearRegression()
#lr.fit(all_features[:1000].fillna(0.0), all_ys[:1000])


# +
# best_model
# -

# # Compare Model 6
# Final Model

# fold = 10, train_size = 0.98, polynomial_features = True,normalize_method = 'minmax',

best_model = compare_models(fold = 10, sort = 'F1', n_select = 3 )

# # Create Models

# +
# Extra tree classifier
# double check folds
et = create_model('et', fold = 10)

# Random Forest 
rf = create_model('rf')

# Decision Tree Classifier
dt = create_model('dt')
# -

models()

# # Tune Hyperparam

tuned_et = tune_model(et)


boosted_et = ensemble_model(et, method = 'Boosting')


tuned_rf = tune_model(rf)

boosted_dt = ensemble_model(dt, method = 'Boosting')

# +
# Blend Models

blender = blend_models(estimator_list = [boosted_dt, tuned_et, tuned_rf], method = 'soft')

# +
# Stack Models

# stacker = stack_models(estimator_list = [boosted_dt, tuned_et, tuned_rf], meta_model=rf)

# -

# # Analyze Model

plot_model(rf)

plot_model(tuned_rf)

plot_model(et)

plot_model(tuned_et)

plot_model(rf, plot = 'confusion_matrix')

plot_model(et, plot = 'feature')

plot_model(rf, plot = 'feature')

plot_model(dt, plot = 'feature')

plot_model(rf, plot = 'pr')

plot_model(rf, plot = 'class_report')

evaluate_model(rf)

# # autoML

# +
# best = automl(optimize = 'Recall')
# best
# -

# # Predict

pred_holdouts_rf = predict_model(rf)
pred_holdouts.head()

pred_holdouts_et = predict_model(et)
pred_holdouts.head()

pred_holdouts_dt = predict_model(dt)
pred_holdouts.head()

new_data = all_data.copy()
new_data.drop(['label','id'], axis=1, inplace=True)
predict_new = predict_model(tuned_et, data=new_data)
print(predict_new.shape)
predict_new.head()

predict_new = predict_model(et, data=new_data)
print(predict_new.shape)
predict_new.head()


predict_new = predict_model(tuned_rf, data=new_data)
print(predict_new.shape)
predict_new.head()

predict_new = predict_model(boosted_dt, data=new_data)
print(predict_new.shape)
predict_new.head()

# # Save and Load Model

save_model(rf, model_name='random-forest')

save_model(et, model_name='extra_tree')

save_model(dt, model_name='decision_tree')

save_model(boosted_et, model_name='boosted_extra_tree')

save_model(tuned_rf, model_name='tuned_random_forest')

save_model(boosted_dt, model_name='boosted_decision_tree')

# +
# Loading a model

# loaded_bestmodel = load_model('tuned_random_forest')
# print(loaded_bestmodel)
# -












