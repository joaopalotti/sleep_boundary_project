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

# # Sleep Metrics Calculation

# ALERT: notebook under construction!

# +
# %load_ext autoreload
# %autoreload 2
    
import pandas as pd
import numpy as np
import os
from glob import glob


# imports
from hypnospy import Wearable, Experiment
from hypnospy.data import RawProcessing
from hypnospy.analysis import SleepMetrics

from Ranged_Based_Precision_Recall import get_ranges, recall, precision, f1


# +
def read_file(filename):
    df = pd.read_csv(filename)
    df["sleep"] = df["stages"] > 0
    df["linetime"] = pd.to_datetime(df["linetime"])
    return df

def read_all_files(path):
    input_files = glob(path)
    dfs = []
    for file in input_files:
        dfs.append(read_file(file))
    return pd.concat(dfs)

def get_modelname_from_file(filename):
    model = os.path.basename(filename).split("_predictions")[0]
    model = model.split("sleep_ml_")[1]
    return model


# +
# Read all raw files:
raw_df = read_all_files("../data/Processed_Mesa_gt_WithandWithout_tolerance/*.csv.gz")

# Read test file (this step can be removed if pycaret and other models save the gt_time and pids)
#df_test = pd.read_csv("./test_data.csv.gz")
# -


# Predictions from the ML models

# +
dfs = []
for filename in glob("../data/ml_predictions/*_predictions_it100.csv.gz"):
    model = get_modelname_from_file(filename)
    dftmp = pd.read_csv(filename)[["Label"]]
    dftmp.columns = [model]
    dfs.append(dftmp)
    
pred_ml = pd.concat([df_test[["pid", "ground_truth", "gt_time"]], *dfs], axis=1)
# -


# Predictions from the NN model

dfnn = pd.read_csv("./predictions_nn.csv.gz").rename(columns={"0":"lstm"})
dfnn["lstm"] = dfnn["lstm"].astype(np.bool)

dfpredictions = pd.merge(pred_ml, dfpredictions)

# Merge all files
#raw_df["hyp_time_col"] = pd.to_datetime(raw_df["hyp_time_col"])
#dfs["gt_time"] = pd.to_datetime(dfs["gt_time"])
dfmerged = pd.merge(raw_df, dfpredictions, left_on=["hyp_time_col", "mesaid"], right_on=["gt_time", "pid"])
print("Unique PIDs for test: %d " % len(dfmerged["pid"].unique()))


dfmerged.to_csv("df_to_sleep_metrics.csv.gz")




exp = Experiment()
exp.wearableSetFromFile("/home/palotti/github/sleep_boundary_project/src/df_to_sleep_metrics.csv.gz",
                         cols_for_activity=["activity"], col_for_datetime="linetime", strftime="%Y-%m-%d %H:%M:%S",
                         col_for_pid="pid"
                       )
exp.change_start_hour_for_experiment_day(15)
exp.set_freq_in_secs(30)

# + code_folding=[26]
sm = SleepMetrics(exp)

tmp = []

# Ground Truth_5min:
df_sm = sm.get_sleep_quality("totalSleepTime", wake_sleep_col="ground_truth_5min",
                             outputname= "TST", normalize_per_hour=False)
df_sm["Alg"] = "GroundTruth_5min"
tmp.append(df_sm)



# Other algorithms:
for variant in ["", "Rescored"]:
    for alg in ["Sadeh", "Sazonov", "ColeKripke", "Oakley10", "ScrippsClinic"]:
        alg = variant + alg
        
        df_sm = sm.get_sleep_quality("totalSleepTime", wake_sleep_col="SleepWindow%s" % (alg),
                                     outputname= "TST", normalize_per_hour=False)
        df_sm["Alg"] = alg
        tmp.append(df_sm)

        
        
for alg in ['lda', 'catboost', 'lightgbm', 'rf', 'lr', 'et', 'lstm']:
    df_sm = sm.get_sleep_quality("totalSleepTime", wake_sleep_col=alg,
                 #sleep_period_col=alg,
                 outputname= "TST", 
                 normalize_per_hour=False)
    df_sm["Alg"] = alg
    tmp.append(df_sm)
    

df_sm = pd.concat(tmp)
df_sm
# -

df_sm.groupby("Alg")["TST"].mean()

# +
# In case we want to see the results:
from hypnospy.analysis import Viewer
wid = 11

v = Viewer(exp.get_all_wearables()[wid])
v.view_signals(signal_categories=["sleep"],
               sleep_cols=["ground_truth_5min", "lstm", "RescoredScrippsClinic"],
               #signal_as_area=["ground_truth", "SleepWindowScrippsClinic"],
               #alphas={'ground_truth': 0.3, "SleepWindowScrippsClinic": 0.2},
               )


# -

def evaluate_precision_recall_time_biased(gt_col="ground_truth_5min",
                                          other_col="lstm",
                                          alpha_r = 0.99,
                                          gamma_kind = "default",
                                          bias_kind_p = "flat",
                                          bias_kind_r = "front"):

    rs, ps, f1s = [], [], []
    for e, w in enumerate(exp.get_all_wearables()):
        df = w.data

        ranges_real = get_ranges(df, gt_col)
        ranges_pred = get_ranges(df, other_col)



        r = recall(ranges_real, ranges_pred, alpha_r, bias_kind_r, gamma_kind)
        p = precision(ranges_real, ranges_pred, alpha_r, bias_kind_p, gamma_kind)
        f = f1(ranges_real, ranges_pred, alpha_r, bias_kind_r, bias_kind_p, gamma_kind)
        #print("P: %.6f, R: %.6f, F1: %.6f" % (p, r, f))
        ps.append(p)
        rs.append(r)
        f1s.append(f)
        
    return np.mean(ps), np.mean(rs), np.mean(f1s)

evaluate_precision_recall_time_biased("ground_truth_5min", "lstm")

evaluate_precision_recall_time_biased("ground_truth_5min", "Sadeh")

evaluate_precision_recall_time_biased("ground_truth_5min", "RescoredScrippsClinic")

for alg in ['Sadeh', 'Sazonov', 'ColeKripke', 'Oakley10', 'ScrippsClinic', 'RescoredSadeh', 
            'RescoredSazonov', 'RescoredColeKripke', 'RescoredOakley10', 'RescoredScrippsClinic', 
            'lda', 'catboost', 'lightgbm', 'rf', 'lr', 'et', 'lstm']:
    print("Alg:", alg, ":", evaluate_precision_recall_time_biased("ground_truth_5min", alg))
