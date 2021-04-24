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
pd.set_option('display.max_rows', 100)

import numpy as np
import os
from tqdm import tqdm
from glob import glob
from datetime import timedelta
import datetime


# -

def get_ranges(df, col):
    temp = []
    return_array = []
    for i in df.index:
        if(df.loc[i, col] == True):
            temp.append(i)

            if i+1 in df.index and df.loc[i+1, col] != True:
                return_array.append(temp)
                temp = []
                continue;
    
    if len(temp) > 0:
        return_array.append(temp)
    return return_array


# +
def positional_bias(i, anomaly_len, kind = "flat"):
    anomaly_len -= 1
    if(kind == "flat"):
        return 1
    if(kind == "front"):
        return anomaly_len - i + 1
    if(kind == "back"):
        return i
    if(kind == "middle"):
        if(i <= anomaly_len/2):
            return i
        else:
            return anomaly_len - i + 1

def w(period, overlap_set, bias_kind = "flat"):
    max_val = 0
    my_val = 0
    anomaly_len = len(period)
    
    for i in range(len(period)):
        bias = positional_bias(i, anomaly_len, kind = bias_kind)
        max_val = max_val + bias
        if(period[i] in overlap_set):
            my_val = my_val + bias
            
    return 0 if max_val == 0 else my_val/max_val



# +
def gamma(x, kind = "default"):
    if (kind == "default" or kind =="one"):
        return 1
    if (kind == "reciprocal"):
        return 1/x
    

def overlap(x, y):
    return list(set(x) & set(y))

def cardinality_factor(period_x, range_of_periods_y, gamma_kind):
    x = 0
    if(period_x in range_of_periods_y): 
        return 1
    else:
        for predict in range_of_periods_y:
            if(all(i in predict for i in period_x)): #if real boundary is a subset of prediction boundary
                return 1
            if (any(i in period_x for i in predict) ):
                x += 1
                
        if(x == 0):
            return 0
        else:
            return gamma(x, gamma_kind)


def overlap_reward(period_x, range_of_periods_y, bias_kind = "flat", gamma_kind = "default"):
    cardinality = cardinality_factor(period_x, range_of_periods_y, gamma_kind = gamma_kind)
    sum_weight = 0
    
    for predict in range_of_periods_y:
        
        overlap_set = overlap(predict,period_x) 
        weight = w(period_x, overlap_set, bias_kind)
        sum_weight = sum_weight + weight 
        
    return cardinality*sum_weight


def existence_reward(period_x, range_of_periods_y):
    all_predict = []
    for i in range(len(range_of_periods_y)):
        all_predict.extend(range_of_periods_y[i])
        
    if (any(i in period_x for i in all_predict) ):
        return 1
    else:
        return 0


# -

def recall(real_anomaly_range, predicted_anomaly_ranges, alpha= 0, bias_kind = "flat", gamma_kind="reciprocal"):
    total_recall = 0 
    for r in real_anomaly_range:
        recall_t = ( alpha*existence_reward(r, predicted_anomaly_ranges) ) + ( (1-alpha)*overlap_reward(r, predicted_anomaly_ranges, bias_kind, gamma_kind))     
        total_recall = total_recall + recall_t
        
    return total_recall/len(real_anomaly_range)


def precision(real_anomaly_range, predicted_anomaly_ranges, alpha= 0, bias_kind = "flat", gamma_kind="reciprocal"):
    total_precision = 0
    for p in predicted_anomaly_ranges:
        precision_t = overlap_reward(p, real_anomaly_range, bias_kind )
        total_precision = total_precision + precision_t
        
    return total_precision/len(predicted_anomaly_ranges)


def f1(real_anomaly_range, predicted_anomaly_ranges, alpha= 0, bias_kind_r = "flat", bias_kind_p = "flat", gamma_kind="reciprocal"):
#    Assumin beta = 1
    r = recall(real_anomaly_range, predicted_anomaly_ranges, alpha, bias_kind_r, gamma_kind)
    p = precision(real_anomaly_range, predicted_anomaly_ranges, alpha, bias_kind_p, gamma_kind)
    f1 = 2 * ( (r*p) / (r+p))
    return f1

# +
# simple
#df_pred = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/simple/simple.pred")
#df_real = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/simple/simple.real")

# #ecg
# df_pred = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/ecg/lstm_ad.pred")
# df_real = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/ecg/lstm_ad.real")

#aapl
#df_pred = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/aapl/lstm_ad.pred")
#df_real = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/aapl/lstm_ad.real")

#NYC_taxi
# df_pred = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/nyc_taxi/luminol.pred")
# df_real = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/nyc_taxi/luminol.real")

# ranges_pred = get_ranges(df_pred, "0")
# ranges_real = get_ranges(df_real, "0")

# +
# # Tests that failed 1
# df_pred = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/nyc_taxi/luminol.pred")
# df_real = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/nyc_taxi/luminol.real")

# ranges_pred = get_ranges(df_pred, "0")
# ranges_real = get_ranges(df_real, "0")

# alpha_r = 0.99
# gamma_kind = "reciprocal"
# bias_kind_p = "back"
# bias_kind_r = "back"

# r = recall(ranges_real, ranges_pred, alpha_r, bias_kind_r, gamma_kind)
# p = precision(ranges_real, ranges_pred, alpha_r, bias_kind_p, gamma_kind)
# f = f1(ranges_real, ranges_pred, alpha_r, bias_kind_r, bias_kind_p, gamma_kind)
# print("P: %.6f, R: %.6f, F1: %.6f" % (p, r, f))

# +
# # Tests that failed 2
# df_pred = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/simple/simple.pred")
# df_real = pd.read_csv("https://raw.githubusercontent.com/IntelLabs/TSAD-Evaluator/master/examples/simple/simple.real")

# ranges_pred = get_ranges(df_pred, "0")
# ranges_real = get_ranges(df_real, "0")

# alpha_r = 0.99
# gamma_kind = "one"
# bias_kind_p = "flat"
# bias_kind_r = "flat"

# r = recall(ranges_real, ranges_pred, alpha_r, bias_kind_r, gamma_kind)
# p = precision(ranges_real, ranges_pred, alpha_r, bias_kind_p, gamma_kind)
# f = f1(ranges_real, ranges_pred, alpha_r, bias_kind_r, bias_kind_p, gamma_kind)
# print("P: %.6f, R: %.6f, F1: %.6f" % (p, r, f))


# # 
# # ~/github/TSAD-Evaluator/src]$ ./evaluate -t ../examples/simple/simple.real ../examples/simple/simple.pred 1 0.99 one flat flat
# # Precision = 0.21875
# # Recall = 0.665
# # F-Score = 0.329208
# # Precision is failing for this test

# -




