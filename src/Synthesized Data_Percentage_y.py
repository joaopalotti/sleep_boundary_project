# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
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

# +
# Make a synthesized dataframe for ground truth

df_PSG = pd.DataFrame({
    'linetime': pd.date_range(start='2017-01-04 1:00:00', end = '2017-01-04 3:00:00', freq='30S')
})
df = pd.DataFrame({
    'linetime': pd.date_range(start='2017-01-04 1:00:00', end = '2017-01-04 3:00:00', freq='30S')
})

"""
# Add labels wake:0/sleep:1
# Adding 2 sleep periods [1:30_1:50, 2:30_2:45]
"""

# df_PSG ['w/s'] = 0

# i = 60
# j = 100
# for c in range(i, j+1):
#     df_PSG.at[c, 'w/s'] = 1
    
# i = 180
# j = 210
# for c in range(i, j+1):
#     df_PSG.at[c, 'w/s'] = 1
    
# df_PSG.loc[99:103]


df['w/s'] = 0
i = 60
j = 110
for c in range(i, j+1):
    df.at[c, 'w/s'] = 1
    
i = 150
j = 210
for c in range(i, j+1):
    df.at[c, 'w/s'] = 1
    
df.loc[99:103]

# +
# import matplotlib.pyplot as plt

# plt.plot(df_PSG['linetime'], df_PSG['w/s'])
# -

plt.plot(df['linetime'], df['w/s'])


# Generate slide windows
def generate_slide_wins(df, winsize=21):
    
    seq_id = 0
    transformed_df = []
    list_of_indexes=[] 
#     labels_5min = []
    labels = []
    linetime = []
#     activity = []
    
    df.index.to_series().rolling(winsize, center=True).apply((lambda x: list_of_indexes.append(x.tolist()) or 0), raw=False)
    
    
    for idx in list_of_indexes:
        # to Do: this column name "ground truth" shuld be changed later
#         labels_5min.append(df.iloc[idx].iloc[winsize//2]["ground_truth_5min"]) #take the middle value of the window 
        labels.append(df.iloc[idx].iloc[winsize//2]["w/s"]) #take the middle value of the window
        linetime.append(df.iloc[idx].iloc[winsize//2]["linetime"])   #we need this for sleep metrics calculation later
#         activity.append(df.iloc[idx].iloc[winsize//2]["activity"])   #we need this for sleep metrics calculation later
        
        tmp_df = df.iloc[idx].copy()
        tmp_df["seq_id"] = seq_id
        seq_id += 1
        transformed_df.append(tmp_df)

    return pd.concat(transformed_df), pd.Series(labels),pd.Series(linetime)

transformed_df, labels, linetime = generate_slide_wins(df, 11)

labels[45:70]

labels.shape

linetime[45:70]

transformed_df[550:600]

# +
# Now we want to see what percentage of each boundary is detected as sleep(not sure)
temp = transformed_df.groupby(['seq_id'])['w/s'].sum()


window_size = 11
percentages = []
for i in temp:
    #calculate percentage
    percentage = 100*i/window_size
    percentages.append(percentage)

# -

linetime = linetime.to_frame()
linetime['percentage'] = percentages
# linetime

linetime[40:70]




