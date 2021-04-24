# ---
# jupyter:
#   jupytext:
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
from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters
import itertools
from pycaret.classification import *
import tsfresh
import re
from pandas.api.types import is_datetime64_ns_dtype

from sklearn.model_selection import KFold


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


# -


def generate_slide_wins(df_in, start_seq, winsize, delta,
                        time_col="time", label_col="ground_truth", pid_col="mesaid",
                        dataset_freq="0h00t30s"):
    """
    Expected a timeseries data from ONE participant only.
    We used the first hour to extract the features and the 15 min only to collect the ground_truth labels.
    """
    seq_id = start_seq
    transformed_df = []
    list_of_indices = []
    df_labels = []
    df_label_times = []

    if pid_col not in df_in:
        print("PID col not found")
        return
    
    if time_col not in df_in:
        print("Time col not found")
        return
    
    pid = df_in[pid_col].unique()
    if len(pid) > 1:
        print("ERROR: We should have only one pid here. Aborting")
        return
    pid = pid[0]
    
    df = df_in.reset_index(drop=True).copy()
    
    if not is_datetime64_ns_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    
    # The following code will construct a rolling win that could be based on either time or #win
    # This will feed list_of_indexes with the sub-win indices that will be used in the next for loop
    df.reset_index().rolling(winsize, on=time_col, center=False, closed="both")["index"].apply((lambda x: list_of_indices.append(x.tolist()) or 0))

    # Time-based win might be smaller than the expected size. 
    expected_size = (winsize // pd.Timedelta(dataset_freq)) + 1

    # Save a time indexed copy of the dataframe
    dftime = df.set_index(time_col).copy()
    
    for idx in list_of_indices:
        if len(idx) != expected_size:
            continue
            
        last_row = df.loc[idx].iloc[-1]
        label_time = last_row[time_col] + pd.Timedelta(delta)
        
        if label_time not in dftime.index:
            continue
        
        label = dftime.loc[label_time][label_col]
        
        # save instance
        df_labels.append(label)
        df_label_times.append(label_time)
        
        tmp_df = df.loc[idx].copy()
        tmp_df["seq_id"] = seq_id
        seq_id += 1

        del tmp_df[pid_col]
        
        transformed_df.append(tmp_df)

    df_labels = pd.Series(df_labels)
    df_labels.name = "ground_truth"
    
    df_label_times = pd.Series(df_label_times)
    df_label_times.name = "gt_time"
    
    transformed_df = pd.concat(transformed_df).reset_index(drop=True)
    df_pid = pd.Series([pid] * df_labels.shape[0])
    df_pid.name = "pid"
    
    return seq_id, transformed_df, df_labels, df_label_times, df_pid


def generate_timeseries_df(df, signals, winsize, delta, label_col, time_col, pid_col):

    df_labels = []
    df_label_times = []
    df_timeseries = []
    df_pids = []

    last_seq_id = 0

    for pid in tqdm(sorted(df[pid_col].unique())):
        df_tmp = df[df[pid_col] == pid]
        last_seq_id, df_ts, df_label, df_label_time, df_pid = generate_slide_wins(df_tmp[signals],
                                                                                  start_seq=last_seq_id,
                                                                                  winsize=winsize,
                                                                                  delta=delta,
                                                                                  label_col=label_col,
                                                                                  pid_col=pid_col,
                                                                                  time_col=time_col)
        df_timeseries.append(df_ts)
        df_labels.append(df_label)
        df_label_times.append(df_label_time)
        df_pids.append(df_pid)

    df_timeseries = pd.concat(df_timeseries).reset_index(drop=True)
    df_labels = pd.concat(df_labels).reset_index(drop=True).to_frame()
    df_label_times = pd.concat(df_label_times).reset_index(drop=True).to_frame()
    
    df_pids = pd.concat(df_pids).reset_index(drop=True).to_frame()
    df_pids.name = "pid"
    
    return df_timeseries, df_labels, df_label_times, df_pids


# +
def data_exists(datafolder):
    if not os.path.exists(datafolder):
        print("Path does not exist")
        return False
    
    for f in ["transformed_df.csv.gz", "df_labels.csv.gz",
              "df_label_times.csv.gz", "df_pids.csv.gz"]:
        if not os.path.exists("%s/%s" % (datafolder, f)):
            print("File %s does not exist" % (f))
            return False
        
    return True

def save_data(output_folder, transformed_df, df_labels, label_times, pids):
    transformed_df.to_csv(os.path.join(output_folder, "transformed_df.csv.gz"), index=False)
    df_labels.to_csv(os.path.join(output_folder, "df_labels.csv.gz"), index=False)
    df_label_times.to_csv(os.path.join(output_folder, "df_label_times.csv.gz"), index=False)
    df_pids.to_csv(os.path.join(output_folder, "df_pids.csv.gz"), index=False)

def load_data(datafolder):
    transformed_df = pd.read_csv("../data/feature_extraction/transformed_df.csv.gz")
    df_labels = pd.read_csv("../data/feature_extraction/df_labels.csv.gz", squeeze=True)
    df_label_times = pd.read_csv("../data/feature_extraction/df_label_times.csv.gz")
    df_pids = pd.read_csv("../data/feature_extraction/df_pids.csv.gz")
    return transformed_df, df_labels, df_label_times, df_pids


# -

def get_raw_features(df, raw_cols, seq_id_col="seq_id"):
    """
    THe df input here is the ``transformed_df`` from the generate_timesiries_df.
    The raw_cols are features such as 'activity' and/or 'mean_hr'
    """

    df["tcounter"] = 1
    df["tcounter"] = df.groupby(seq_id_col)["tcounter"].cumsum()

    pivoted_df = df.pivot(index=seq_id_col, columns="tcounter", values=raw_cols)
    pivoted_df.columns = ["_".join(map(str, c)) for c in pivoted_df.columns]
    
    return pivoted_df


# define a setting for feature Extraction: {Minimal+Fourieh}
ext_settings = {
     "activity": {"sum_values": None,
                  "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
                  "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
                  "root_mean_square": None,
                  "maximum": None,
                  "minimum": None,
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


# +
# Convert time to sin_time, cos_time
def convert_time_sin_cos(df, datetime_col):

    if datetime_col not in df:
        print("ERROR datetime_col %s not in df!" % (datetime_col))
        return

    if not is_datetime64_ns_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    day = 24*60*60
    
    ts = df[datetime_col].apply(lambda x: x.timestamp()).astype(int)
    day_sin = np.sin(ts * (2 * np.pi / day))
    day_cos = np.cos(ts * (2 * np.pi / day))
    
    day_sin.name = "time_sin"
    day_cos.name = "time_cos"
    return day_sin, day_cos


# -

def map_id_fold(all_ids, n): 
    pids = all_ids["pid"].unique().ravel()
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    mapping = []
    for i, (_, test) in enumerate(kf.split(pids)):
        for pid_index in test:
            mapping.append({'fold': i, 'pid': pids[pid_index]})

    return pd.DataFrame(mapping)



# +
def save_train_test_splits(output_path, data, label):
    
    test_data = data[data["fold"] == 10] # handout and never used in the training
    train_data = data[data["fold"] != 10] 

    print("Total number of users: %d" % data["pid"].unique().shape[0])
    print("In the training set: %d" % train_data["pid"].unique().shape[0])
    print("In the test set: %d" %  test_data["pid"].unique().shape[0])
    
    # To remove Json Error from pycarot's setup function
    data = raw_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    train_data = train_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    test_data = test_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    data.to_csv(os.path.join(output_path, "all_%s_data.csv.gz" % (label)), index=False)
    train_data.to_csv(os.path.join(output_path, "train_%s_data.csv.gz" % (label)), index=False)
    test_data.to_csv(os.path.join(output_path, "test_%s_data.csv.gz" % (label)), index=False)

    
def convert_str_time(t):
    if t < 0:
        return "-" + convert_str_time(-1*t)
    h = t // 60
    m = t % 60
    return "%dh%01dt" % (h,m)


# +

for winsize_in_minutes in [10, 20, 40]:
    for centered in [True, False]:

    centered_str = "centered" if centered else "notcentered"
    winsize_str = convert_str_time(winsize_in_minutes)
    delta_str = convert_str_time(-winsize_in_minutes//2) if centered else "0h00t"

    train_test_output_path = "../data/processed/train_test_splits/%smin_%s" % (winsize_in_minutes, centered_str)
    feature_extration_datapath = "../data/feature_extraction_%dm_%s/" % (winsize_in_minutes, centered_str)

    if not os.path.exists(feature_extration_datapath):
        os.mkdir(feature_extration_datapath)


    if not data_exists(feature_extration_datapath):
        df = read_all_files("../data/Processed_Mesa_gt_WithandWithout_tolerance/*.csv.gz")
        df["hyp_time_col"] = pd.to_datetime(df["hyp_time_col"])
        win_result = generate_timeseries_df(df, signals=["ground_truth_5min", "mesaid", "hyp_time_col", "activity", "mean_hr"],
                                            winsize=winsize_str, 
                                            delta=delta_str,
                                            label_col="ground_truth_5min", pid_col="mesaid", 
                                            time_col="hyp_time_col")

        transformed_df, df_labels, df_label_times, df_pids = win_result
        save_data(feature_extration_datapath, transformed_df, df_labels, df_label_times, df_pids)

    else:
        transformed_df, df_labels, df_label_times, df_pids = load_data(feature_extration_datapath)


    # Extract raw features
    raw_features = get_raw_features(transformed_df, ["activity", "mean_hr"], "seq_id")

    feature_extracted = os.path.join(feature_extration_datapath, "extracted_features.csv.gz")

    if not os.path.exists(feature_extracted):
        print("Extracting features...")
        extracted_features = tsfresh.extract_relevant_features(transformed_df[["activity", "mean_hr", "hyp_time_col", "seq_id"]],
                                                               df_labels["ground_truth"],
                                                               column_id="seq_id", column_sort="hyp_time_col",
                                                               disable_progressbar=True,
                                                               default_fc_parameters={}, kind_to_fc_parameters=ext_settings)
        extracted_features.to_csv(feature_extracted, index=False)

    else:
        print("Reading extracted features from file '%s'..." % (feature_extracted))
        extracted_features = pd.read_csv(feature_extracted)


    df_pid_fold = map_id_fold(df_pids, 11)  
    df_pid_fold = df_pids.merge(df_pid_fold) 

    df_time_sin, df_time_cos = convert_time_sin_cos(df_label_times, "gt_time")

    tsfresh_data = pd.concat([df_pid_fold.reset_index(drop=True),
                              df_label_times.reset_index(drop=True),
                              df_labels.reset_index(drop=True),
                              extracted_features.reset_index(drop=True), 
                              df_time_sin.reset_index(drop=True), df_time_cos.reset_index(drop=True)], axis=1)

    raw_data = pd.concat([df_pid_fold.reset_index(drop=True),
                          df_label_times.reset_index(drop=True),
                          df_labels.reset_index(drop=True),
                          raw_features.reset_index(drop=True)], axis=1)

    save_train_test_splits(train_test_output_path, tsfresh_data, "tsfresh")
    save_train_test_splits(train_test_output_path, raw_data, "raw")

