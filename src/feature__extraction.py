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
from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters
import itertools
from pycaret.classification import *
import tsfresh

from sklearn.model_selection import KFold


# -

def read_file(filename):
    df = pd.read_csv(filename)
    df["sleep"] = df["stages"] > 0
    df["linetime"] = pd.to_datetime(df["linetime"])
    return df


def generate_slide_wins(df, winsize=11):
    
    seq_id = 0
    transformed_df = []
    list_of_indexes=[] 
    labels_5min = []
    labels_0min = []
    linetime = []
    activity = []
    
    df.index.to_series().rolling(winsize, center=True).apply((lambda x: list_of_indexes.append(x.tolist()) or 0), raw=False)
    
    
    for idx in list_of_indexes:
        # to Do: this column name "ground truth" shuld be changed later
        labels_5min.append(df.iloc[idx].iloc[winsize//2]["ground_truth_5min"]) #take the middle value of the window 
        labels_0min.append(df.iloc[idx].iloc[winsize//2]["ground_truth_0min"]) #take the middle value of the window
        linetime.append(df.iloc[idx].iloc[winsize//2]["linetime"])   #we need this for sleep metrics calculation later
        activity.append(df.iloc[idx].iloc[winsize//2]["activity"])   #we need this for sleep metrics calculation later
        
        tmp_df = df.iloc[idx].copy()
        tmp_df["seq_id"] = seq_id
        seq_id += 1
        transformed_df.append(tmp_df)

    return pd.concat(transformed_df), pd.Series(labels_5min), pd.Series(labels_0min),pd.Series(linetime), pd.Series(activity)


# +
input_files = glob("../data/Processed_Mesa_gt_WithandWithout_tolerance/*.csv.gz")
all_ys_5min = []
all_ys_0min = []
all_features = []
all_ids = []
all_linetime = []
all_activity = []

# Read files and roll 21 windows
for file in input_files:
    df = read_file(file)
    transformed_df, labels_5min, labels_0min, linetime, activity = generate_slide_wins(df, 21)

    
# define a setting for feature Extraction: {Minimal+Fourieh}
    ext_settings = {
             "activity": {"sum_values": None,
                          "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
                          "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
                          "root_mean_square": None,
                          "maximum": None,
                          "minimum": None,
#                         "permutation_entropy": [{"tau": 1, "dimension": x} for x in [5, 6, 7]], 
#                           "activity__matrix_profile__feature_max__threshold_0.98": None,
#                           "activity__matrix_profile__feature_mean__threshold_0.98": None,
#                           "matrix_profile__feature_median__threshold_0.98": None,
#                           "matrix_profile__feature_25__threshold_0.98": None,
#                           "matrix_profile__feature_75__threshold_0.98": None,
                          "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]], 
                          "fft_coefficient": [{"coeff": k, "attr": a} for a, k in
                                itertools.product(["real", "imag", "abs", "angle"], range(10))],
                          "fourier_entropy": [{"bins": x} for x in [2, 3, 5, 10, 100]],
                         },
    
             "mean_hr": {"sum_values": None,
                          "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
                          "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
                          "root_mean_square": None,
                          "maximum": None,
                          "minimum": None,
                          "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]], 
                          "fft_coefficient": [{"coeff": k, "attr": a} for a, k in
                                itertools.product(["real", "imag", "abs", "angle"], range(10))],
                          "fourier_entropy": [{"bins": x} for x in [2, 3, 5, 10, 100]],
                        }
}


    extracted_features = tsfresh.extract_features(transformed_df[["activity", "mean_hr", "linetime", "seq_id"]],
                                            column_id="seq_id",
                                            column_sort="linetime",
                                            impute_function=tsfresh.utilities.dataframe_functions.impute,
                                            disable_progressbar=True,
                                            default_fc_parameters={},
                                            kind_to_fc_parameters=ext_settings)
    impute(extracted_features)
    
    ids = pd.Series(labels_5min.shape[0]*[df["mesaid"].unique()[0]])
    
    all_features.append(extracted_features)
    all_ys_5min.append(labels_5min)
    all_ys_0min.append(labels_0min)
    all_ids.append(ids)
    all_linetime.append(linetime)
    all_activity.append(activity)
    


# +
# # concat all features, ids, and labels
all_ids = pd.concat(all_ids)
all_features = pd.concat(all_features)
all_ys_5min = pd.concat(all_ys_5min)
all_ys_0min = pd.concat(all_ys_0min)
all_linetime = pd.concat(all_linetime)
all_activity = pd.concat(all_activity)


# give name to columns 
all_ids = pd.DataFrame(all_ids)
all_ids.columns = ['id']

all_ys_5min = pd.DataFrame(all_ys_5min)
all_ys_5min.columns = ['label_5min']

all_ys_0min = pd.DataFrame(all_ys_0min)
all_ys_0min.columns = ['label_0min']

all_linetime = pd.DataFrame(all_linetime)
all_linetime.columns = ['linetime']

all_activity = pd.DataFrame(all_activity)
all_activity.columns = ['activity']





# +
# # write into CSVs
# all_features.to_csv("../data/all_features/all_features_win21.csv.gz", index=False)
# all_ys.to_csv("../data/all_features/all_ys_win21.csv.gz", index=False)
# all_ids.to_csv("../data/all_features/all_ids_win21.csv.gz", index=False)
# all_linetime.to_csv("../data/all_features/all_linetime_win21.csv.gz", index=False)
# all_activity.to_csv("../data/all_features/all_activity_win21.csv.gz", index=False)


# read them
# all_features = pd.read_csv("../data/all_features/all_features_win21.csv.gz")
# all_ids = pd.read_csv("../data/all_features/all_ids_win21.csv.gz")
# all_ys = pd.read_csv("../data/all_features/all_ys_win21.csv.gz")
# all_linetime = pd.read_csv("../data/all_features/all_linetime_win21.csv.gz")
# all_activity = pd.read_csv("../data/all_features/all_activity_win21.csv.gz")

# change column names
# all_ids.columns = ['id']
# all_ys.columns = ['label']
# all_linetime.columns = ['linetime']
# all_activity.columns = ['activity']

# +
def map_id_fold(all_ids, n): 
    pids = all_ids["id"].unique().ravel()
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    mapping = []
    for i, (_, test) in enumerate(kf.split(pids)):
        for pid_index in test:
            mapping.append({'fold': i, 'id': pids[pid_index]})

    return pd.DataFrame(mapping)


df_pid_fold = map_id_fold(all_ids, 11)  #Change fold later
df_pid_fold = pd.merge(df_pid_fold, all_ids)

all_data = pd.concat([df_pid_fold.reset_index(drop=True), all_ys_5min.reset_index(drop=True),
                      all_ys_0min.reset_index(drop=True), all_features.reset_index(drop=True), 
                      all_activity.reset_index(drop=True),all_linetime.reset_index(drop=True)] axis=1)



# +
test_data = all_data[all_data["fold"] == 10] # handout and never used in the training
train_data = all_data[all_data["fold"] != 10] 

train_data.shape, test_data.shape
# -


# TODO: need to save the pid of the testset:
test_ids = test_data["id"].unique()
test_ids


print("Total number of users: %d" % all_data["id"].unique().shape[0])
print("In the training set: %d" % train_data["id"].unique().shape[0])
print("In the test set: %d" %  test_data["id"].unique().shape[0])

# +
# to remove Json Error from setup function

import re
all_data = all_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
train_data = train_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
test_data = test_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# test_data
# -

def pycater_setup(gt_label = "label_5min", ignore_feat= ["id", "fold", "linetime", "activity"]):
    
    experiment = setup(data=train_data, test_data=test_data,
                   target=gt_label,
#                    train_size = 0.98, # when the data is bigger than 1M it counts as Big Data
                   session_id=123,
                   normalize=True,
                   transformation=True,
                   fold_strategy="groupkfold",
                   fold_groups="fold",
                   ignore_features= ignore_feat
                   silent=True,
                   use_gpu=False,
#                    normalize_method = 'zscore',
                   normalize_method = 'minmax',       
#                    remove_outliers = True,
                   polynomial_features = True,
#                    fix_imbalance = True,
                   )
    return experiment



# # Setup with 5min tolerance

experiment = pycater_setup(gt_label = "label_5min", ["id", "fold", "linetime", "activity", "label_0min"])

# +
# columns that had been removed by pycaret
# diff = [item for item in train_data.columns if item not in get_config('X_train').columns]
# diff

# +
# et = create_model("et")
# -

best_model_5min = compare_models( fold = 10, sort = 'F1', n_select = 3 )


best_model_5min

# # setup without tolerance

experiment = pycater_setup(gt_label = "label_0min", ["id", "fold", "linetime", "activity", "label_5min"])

best_model_0min = compare_models( fold = 10, sort = 'F1', n_select = 3 )


# # Save predictions

# +
# A loop on top 3 classifiers
predict_5min = {} #this dict contains the predicted result from classifiers with 5min tolerance
                  #key represents the rank of model in best model results
predict_0min = {} #this dict contains the predicted result from classifiers with 0min tolerance
                  #key represents the rank of model in best model results
    
# n = 0 #tuye loop in az 0 ta 2 mire
for n in range(0, 3):
    predict_5min[n] = predict_model(best_model_5min[n], data = test_data)
    predict_0min[n] = predict_model(best_model_0min[n], data = test_data)

    predict_5min[n].reset_index(drop=True, inplace=True)
    predict_0min[n].reset_index(drop=True, inplace=True)
    
    predict_5min[n]['sleep_wake_5min_%sthBest' %(n)] = 0
    predict_0min[n]['sleep_wake_5min_%sthBest' %(n)] = 0

    for i in range(len(predict_5min[n])):
        if (predict_5min[n].loc[i, 'Label'] == 'True'):
            predict_5min[n].at[i, 'sleep_wake_5min_%sthBest' %(n)] = 1 # 0;wake / 1;sleep
    for i in range(len(predict_0min[n])):
        if (predict_0min[n].loc[i, 'Label'] == 'True'):
            predict_0min[n].at[i, 'sleep_wake_0min_%sthBest' %(n)] = 1 # 0;wake / 1;sleep
        
    
    predict_5min[n].drop(predict_5min[n].columns.difference(['id', 'sleep_wake_5min_%sthBest' %(n),'linetime','activity']), axis = 1, inplace=True)   
    predict_0min[n].drop(predict_0min[n].columns.difference(['id', 'sleep_wake_0min_%sthBest' %(n),'linetime','activity']), axis = 1, inplace=True)   

    # write to the disk the result of classifiers
    predict_5min[n].to_csv("../data/ML_Classifiers/predict_5min_best%sth.csv.gz" (n), index=False)
    predict_0min[n].to_csv("../data/ML_Classifiers/predict_0min_best%sth.csv.gz" (n), index=False)
    
# -

# # Sleep Metrics Calculation

# imports
from hypnospy import Wearable, Experiment
from hypnospy.data import RawProcessing
from hypnospy.analysis import NonWearingDetector, SleepBoudaryDetector, Validator, Viewer, SleepWakeAnalysis, SleepMetrics


# +
# sleep metrics from hypnospy
start_hour=15
exp = Experiment()
pp = RawProcessing("../data/ML_Classifiers/predict_5min_best1th.csv.gz", cols_for_activity=["activity"],
                  col_for_datetime="linetime", strftime="%Y-%m-%d %H:%M:%S", col_for_pid="id")
w = Wearable(pp)  
exp.add_wearable(w)
exp.set_freq_in_secs(30)
w.change_start_hour_for_experiment_day(start_hour)
sm = SleepMetrics(w)
# sm


tmp = []
df_sm = sm.get_sleep_quality("totalSleepTime", wake_sleep_col="sleep_wake_5min_1thBest",
#                               sleep_period_col="sleep_wake_col_et",
                              outputname= "TST", normalize_per_hour=False)
df_sm["Alg"] = "5_min_1thBest"
tmp.append(df_sm)

df_sm = pd.concat(tmp)
df_sm

# ERROR: This cell shows the result only for the first id, also it is sth weird! 

# +
# The sleep column shows the sleep epochs. /to get the TST we have to multiply it by 30s

# predict_et.groupby('id').sum()
# -


