#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing packages
import pandas as pd
import math
import numpy as np
import datetime

# Loading data
data = pd.read_csv("新竹_2020.csv", encoding="utf-8")
data


# In[2]:


data.columns


# In[3]:


data["Date"] = pd.to_datetime(data["Date"], format="%Y/%m/%d  %H:%M")
data["Date"]


# In[4]:


date_data = data[(data["Date"] >= "2020-10-01") & (data["Date"] <= "2020-12-31")]
date_data


# In[5]:


nan_data = date_data[
    [
        "Station",
        "Date",
        "Item",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
    ]
].apply(pd.to_numeric, errors="coerce")
nan_data = date_data[["Station", "Date", "Item"]].join(nan_data)
nan_data.at[6582, "9"]


# In[6]:


def fill_missing(data):
    data_filled = data
    for r in data.index:
        for c in range(24):
            if np.isnan(data.at[r, str(c)]):
                # Front
                if c == 0:
                    row = r - 18
                    column = 23
                else:
                    row = r
                    column = c - 1
                front = data.at[row, str(column)]
                while np.isnan(front):
                    if column == 0:
                        row = row - 18
                        column = 23
                    else:
                        row = row
                        column = column - 1
                    front = data.at[row, str(column)]
                # Back
                if c == 23:
                    row = r + 18
                    column = 0
                else:
                    row = r
                    column = c + 1
                if row > 6587:
                    back = front
                back = data.at[row, str(column)]
                while np.isnan(back):
                    if column == 23:
                        row = row + 18
                        column = 0
                        if row > 6587:
                            back = front
                    else:
                        row = row
                        column = column + 1

                # Calculate average
                data_filled.at[r, str(c)] = (front + back) / 2
            else:
                data_filled.at[r, str(c)] = data.at[r, str(c)]
    return data_filled


# In[8]:


filled_data = fill_missing(nan_data)
filled_data


# In[9]:


train = filled_data[(filled_data["Date"] >= "2020-10-1") & (filled_data["Date"] <= "2020-11-30")]
train


# In[10]:


test = filled_data[(filled_data["Date"] >= "2020-12-1") & (filled_data["Date"] <= "2020-12-31")]
test


# In[11]:


# Test set
merged_test = test[0:18]
merged_test = merged_test[
    [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
    ]
]
merged_test.reset_index(drop=True, inplace=True)
for i in range(1, 31):
    small_merge = test[i * 18 : (i + 1) * 18]
    small_merge = small_merge[
        [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
        ]
    ]
    small_merge.reset_index(drop=True, inplace=True)
    merged_test = pd.concat([merged_test, small_merge], axis=1, ignore_index=True)
print(merged_test)


# In[12]:


# Building time series for training set
merged_train = train[0:18]
merged_train = merged_train[
    [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
    ]
]
merged_train.reset_index(drop=True, inplace=True)
for i in range(1, 61):
    small_merge = train[i * 18 : (i + 1) * 18]
    small_merge = small_merge[
        [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
        ]
    ]
    small_merge.reset_index(drop=True, inplace=True)
    merged_train = pd.concat([merged_train, small_merge], axis=1, ignore_index=True)
print(merged_train)


# In[13]:


# PM2.5 is at the 10th column -> column[9]
# Only take PM2.5 feature
# Handling train
training = merged_train[[0, 1, 2, 3, 4, 5, 6]]
training = training.loc[9]
training.reset_index(drop=True, inplace=True)
df_training = training.to_frame().T
for i in range(1, 1458):
    train = merged_train[[i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6]]
    train = train.loc[9]
    train.reset_index(drop=True, inplace=True)
    train = train.to_frame().T
    df_training
