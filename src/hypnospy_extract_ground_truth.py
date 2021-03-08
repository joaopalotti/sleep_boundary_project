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

# # Need to install HypnosPy first
# https://github.com/HypnosPy/HypnosPy

# +
import tempfile
import pandas as pd
pd.set_option('display.max_rows', 100)

import numpy as np
import os
from tqdm import tqdm
from glob import glob
from datetime import timedelta


# Sleep boundaries
from hypnospy import Wearable, Experiment
from hypnospy.data import RawProcessing
from hypnospy.analysis import NonWearingDetector, SleepBoudaryDetector, Validator, Viewer, SleepWakeAnalysis, SleepMetrics


# -

def setup_experiment(file_path, start_hour=15):
    # Configure an Experiment
    exp = Experiment()

    # Iterates over a set of files in a directory.
    # Unfortunately, we have to do it manually with RawProcessing because we are modifying the annotations
    for file in glob(file_path):
        pp = RawProcessing(file,
                           # HR information
                           col_for_hr="mean_hr",
                           # Activity information
                           cols_for_activity=["activity"],
                           # Datetime information
                           col_for_datetime="linetime",
                           strftime="%Y-%m-%d %H:%M:%S",
                           # Participant information
                           col_for_pid="mesaid")

        w = Wearable(pp)  # Creates a wearable from a pp object
        # Invert the two_stages flag. Now True means sleeping and False means awake
        w.data["sleep"] = (w.data["stages"] > 0)
        
        exp.add_wearable(w)
        exp.set_freq_in_secs(30)
        w.change_start_hour_for_experiment_day(start_hour)

    return exp


DATAPATH = "../data/raw/collection_mesa_hr_30_240/*_combined.csv.gz"
exp = setup_experiment(DATAPATH)

# Run several SleepWake algorithms
swa = SleepWakeAnalysis(exp)
swa.run_all_sleep_algorithms(activityIdx="activity", rescoring=True, inplace=True)

# +
sbd = SleepBoudaryDetector(exp)
sbd.detect_sleep_boundaries(strategy="annotation", 
                            output_col="ground_truth",
                            annotation_col="sleep",
                            annotation_merge_tolerance_in_minutes=5,
                            annotation_only_largest_sleep_period=False
                           )

for variant in ["", "Rescored"]:
    for alg in ["Sadeh", "Sazonov", "ColeKripke", "Oakley10", "ScrippsClinic"]:
        alg = variant + alg
        sbd.detect_sleep_boundaries(strategy="annotation", 
                                    output_col="SleepWindow%s" % (alg), 
                                    annotation_col=alg,
                                    annotation_merge_tolerance_in_minutes=5,
                                    annotation_only_largest_sleep_period=False
                                    )

# +
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

def get_average(metric_fnt, exp, gt_col="ground_truth", other_col="SleepWindowSadeh"):
    tmp_acc = []
    for w in exp.get_all_wearables():
        tmp = metric_fnt(w.data[gt_col], w.data[other_col])
        tmp_acc.append(tmp)
    return np.array(tmp_acc)
    
    

for variant in ["", "Rescored"]:
    for alg in ["Sadeh", "Sazonov", "ColeKripke", "Oakley10", "ScrippsClinic"]:
        alg = variant + alg
        m = get_average(matthews_corrcoef, exp, "ground_truth", "SleepWindow%s" % (alg))
        print("Alg: %s, Mean: %.2f" % (alg, m.mean()))
    


# +
sm = SleepMetrics(exp)

tmp = []

# Ground Truth:
df_sm = sm.get_sleep_quality("totalSleepTime", wake_sleep_col="ground_truth",
                              sleep_period_col="ground_truth",
                              outputname= "TST", normalize_per_hour=False)
df_sm["Alg"] = "GroundTruth"
tmp.append(df_sm)

# Other algorithms:
for variant in ["", "Rescored"]:
    for alg in ["Sadeh", "Sazonov", "ColeKripke", "Oakley10", "ScrippsClinic"]:
        alg = variant + alg
        
        df_sm = sm.get_sleep_quality("totalSleepTime", wake_sleep_col="SleepWindow%s" % (alg),
                         sleep_period_col="SleepWindow%s" % (alg),
                         outputname= "TST", 
                         normalize_per_hour=False)
        df_sm["Alg"] = alg
        tmp.append(df_sm)

df_sm = pd.concat(tmp)
df_sm
# -

df_sm.groupby("Alg")["TST"].mean() # -> On average, RescoredScrippsClinic is really good!

# print out results to disk
for w in exp.get_all_wearables():
    pid = "%04d" % int(w.get_pid())
    w.data.to_csv("../data/processed/mesa/%s.csv.gz" % (pid), index=False)

# In case we want to see the results:
v = Viewer(exp.get_all_wearables()[0])
v.view_signals(signal_categories=["hr"],
               signal_as_area=["ground_truth", "SleepWindowScrippsClinic"],
               alphas={'ground_truth': 0.3, "SleepWindowScrippsClinic": 0.2},
               )

