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
# read features

all_features = pd.read_csv("../data/all_features/all_features_win21.csv")
all_ids = pd.read_csv("../data/all_features/all_ids_win21.csv")
all_ys = pd.read_csv("../data/all_features/all_ys_win21.csv")

# change column names
all_ids.columns = ['id']
all_ys.columns = ['label']


# +
def map_id_fold(all_ids, n): 
    pids = all_ids["id"].unique().ravel()
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    mapping = []
    for i, (_, test) in enumerate(kf.split(pids)):
        for pid_index in test:
            mapping.append({'fold': i, 'id': pids[pid_index]})

    return pd.DataFrame(mapping)


df_pid_fold = map_id_fold(all_ids, 3)  #Change fold later
df_pid_fold = pd.merge(df_pid_fold, all_ids)

all_data = pd.concat([df_pid_fold.reset_index(drop=True), all_ys.reset_index(drop=True), all_features.reset_index(drop=True)], axis=1)


# +
from pycaret.classification import *

experiment = setup(data=all_data, 
                   target="label",
                   
                   session_id=123,
                   normalize=True,
                   transformation=True,
                   fold = 3, #Change it later
                   fold_strategy="groupkfold",
                   fold_groups="fold",
                   ignore_features=["id","fold"],
                   silent=True,
                   use_gpu=False,
#                    normalize_method = 'zscore',
                   normalize_method = 'minmax',       
#                    remove_outliers = True,
                   polynomial_features = True,
#                    fix_imbalance = True,
                   )


# +
# columns that had been removed unexpectedly!

diff = [item for item in all_data.columns if item not in get_config('X_train').columns]
diff
# -

# # Final Model
#

best_model = compare_models(fold = 3, sort = 'F1', n_select = 3 )

# # Make a copy of Data for Prediction

new_data = all_data.copy()
new_data.drop(['fold','label'], axis=1, inplace = True)
new_data

# # Extra Tree Classifier

# +
predict_et = predict_model(best_model[0], data = new_data)
predict_et['sleep'] = 0
for i in range(len(predict_et)):
    if (predict_et.loc[i, 'Label'] == 'True'):
        predict_et.at[i, 'sleep'] = 1 # 0;wake / 1;sleep
        
predict_et.drop(predict_et.columns.difference(['id','sleep']), axis = 1, inplace=True)   
# predict_et
# -

plot_model(best_model[0], plot = 'confusion_matrix')

# +
# The sleep column shows the sleep epochs. /to get the TST we have to multiply it by 30s

predict_et.groupby('id').sum()
# -

# # Extra Tree Classifier tuning

# +
# tune et

tuned_et = tune_model(best_model[0])
predict_tune_et = predict_model(tuned_et, data = new_data)
predict_tune_et['sleep'] = 0
for i in range(len(predict_tune_et)):
    if (predict_tune_et.loc[i, 'Label'] == 'True'):
        predict_tune_et.at[i, 'sleep'] = 1 # 0;wake / 1;sleep
        
predict_tune_et.drop(predict_tune_et.columns.difference(['id','sleep']), axis = 1, inplace=True)   
# predict_tune_et

# temp = predict_et[predict_et['id'] == 1]
# temp

# -

predict_tune_et.groupby('id').sum()

# +
# boosting et

boosted_et = ensemble_model(best_model[0], method = 'Boosting')
predict_boost_et = predict_model(boosted_et, data = new_data)
predict_boost_et['sleep'] = 0
for i in range(len(predict_boost_et)):
    if (predict_boost_et.loc[i, 'Label'] == 'True'):
        predict_boost_et.at[i, 'sleep'] = 1 # 0;wake / 1;sleep
        
predict_boost_et.drop(predict_boost_et.columns.difference(['id','sleep']), axis = 1, inplace=True)   
# predict_boost_et
# -

predict_boost_et.groupby('id').sum()

# # Random Forest

# +
predict_rf = predict_model(best_model[1], data = new_data)
predict_rf['sleep'] = 0
for i in range(len(predict_rf)):
    if (predict_rf.loc[i, 'Label'] == 'True'):
        predict_rf.at[i, 'sleep'] = 1 # 0;wake / 1;sleep
        
predict_rf.drop(predict_rf.columns.difference(['id','sleep']), axis = 1, inplace=True)   

# -

plot_model(best_model[1], plot = 'confusion_matrix')

# # Random Forest Tuning

# +
tuned_rf = tune_model(best_model[1])
predict_tune_rf = predict_model(tuned_rf, data = new_data)
predict_tune_rf['sleep'] = 0
for i in range(len(predict_tune_rf)):
    if (predict_tune_rf.loc[i, 'Label'] == 'True'):
        predict_tune_rf.at[i, 'sleep'] = 1 # 0;wake / 1;sleep
        
predict_tune_rf.drop(predict_tune_rf.columns.difference(['id','sleep']), axis = 1, inplace=True)   
# -

predict_tune_rf.groupby('id').sum()

# +
# boosting rf

boosted_rf = ensemble_model(best_model[1], method = 'Boosting')
predict_boost_rf = predict_model(boosted_rf, data = new_data)
predict_boost_rf['sleep'] = 0
for i in range(len(predict_boost_rf)):
    if (predict_boost_rf.loc[i, 'Label'] == 'True'):
        predict_boost_rf.at[i, 'sleep'] = 1 # 0;wake / 1;sleep
        
predict_boost_rf.drop(predict_boost_rf.columns.difference(['id','sleep']), axis = 1, inplace=True)   

# -

predict_boost_rf.groupby('id').sum()

# # Decision tree Classifier

# +
predict_dt = predict_model(best_model[2], data = new_data)
predict_dt['sleep'] = 0
for i in range(len(predict_dt)):
    if (predict_dt.loc[i, 'Label'] == 'True'):
        predict_dt.at[i, 'sleep'] = 1 # 0;wake / 1;sleep
        
predict_dt.drop(predict_dt.columns.difference(['id','sleep']), axis = 1, inplace=True)   
# -

plot_model(best_model[2], plot = 'confusion_matrix')

# # DT Tuning

# +
tuned_dt = tune_model(best_model[2])
predict_tune_dt = predict_model(tuned_dt, data = new_data)
predict_tune_dt['sleep'] = 0
for i in range(len(predict_tune_dt)):
    if (predict_tune_dt.loc[i, 'Label'] == 'True'):
        predict_tune_dt.at[i, 'sleep'] = 1 # 0;wake / 1;sleep
        
predict_tune_dt.drop(predict_tune_dt.columns.difference(['id','sleep']), axis = 1, inplace=True)   
# -

predict_tune_dt.groupby('id').sum()

# +
# boosting rf

boosted_dt = ensemble_model(best_model[2], method = 'Boosting')
predict_boost_dt = predict_model(boosted_dt, data = new_data)
predict_boost_dt['sleep'] = 0
for i in range(len(predict_boost_dt)):
    if (predict_boost_dt.loc[i, 'Label'] == 'True'):
        predict_boost_dt.at[i, 'sleep'] = 1 # 0;wake / 1;sleep
        
predict_boost_dt.drop(predict_boost_dt.columns.difference(['id','sleep']), axis = 1, inplace=True)   

# -

predict_boost_dt.groupby('id').sum()


