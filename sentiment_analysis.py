#!/usr/bin/env python
# coding: utf-8

# In[1]:


#coding = utf-8
#loading package
import pandas as pd
import math
import numpy as np
import datetime

#loading data
data = pd.read_csv("新竹_2020.csv",encoding = "utf-8")
data


# In[2]:


data.columns


# In[3]:


data["日期"]= pd.to_datetime(data["日期"],format ="%Y/%m/%d  %H:%M")
data["日期"]


# In[4]:


date_data = data[(data["日期"]>="2020-10-01")&(data["日期"]<="2020-12-31")]
date_data


# In[5]:


nan_data = date_data[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']].apply(pd.to_numeric,errors = "coerce")
nan_data = date_data[["測站","日期","測項"]].join(nan_data)
nan_data.at[6582,"9"]


# In[6]:


#只是用來測試下面的fuction
test = nan_data
for r in nan_data.index:
        for c in range(24):
            if np.isnan(nan_data.at[r,str(c)]) :
                #front
                if c == 0 :
                    row = r-18
                    column = 23
                else:
                    row = r
                    column = c-1
                front = nan_data.at[row,str(column)]
                while np.isnan(front):
                    if column == 0 :
                        row = row-18
                        column = 23
                    else:
                        row = row
                        column = column-1
                    front = nan_data.at[row,str(column)]
                #back
                if c ==23:
                    row = r+18
                    column = 0
                else:
                    row = r
                    column = c+1
                if row > 6587:
                    back = front
                else :
                    back = nan_data.at[row,str(column)]
                while np.isnan(back):
                    if column ==23:
                        row = row+18
                        column = 0
                        if row > 6587:
                            back = front
                    else:
                        row = row
                        column = column+1
                    
                #取平均
                test.at[r,str(c)]= (front+back)/2
            else:
                test.at[r,str(c)] = nan_data.at[r,str(c)]


# In[7]:


def fill_missing(data):
    data_filled = data
    for r in data.index:
        for c in range(24):
            if np.isnan(data.at[r,str(c)]) :
                #front
                if c == 0 :
                    row = r-18
                    column = 23
                else:
                    row = r
                    column = c-1
                front = data.at[row,str(column)]
                while np.isnan(front):
                    if column == 0 :
                        row = row-18
                        column = 23
                    else:
                        row = row
                        column = column-1
                    front = data.at[row,str(column)]
                #back
                if c ==23:
                    row = r+18
                    column = 0
                else:
                    row = r
                    column = c+1
                if row > 6587:
                    back = front
                back = data.at[row,str(column)]
                while np.isnan(back):
                    if column ==23:
                        row = row+18
                        column = 0
                        if row > 6587:
                            back = front
                    else:
                        row = row
                        column = column+1
                    
                #取平均
                data_filled.at[r,str(c)]= (front+back)/2
            else:
                data_filled.at[r,str(c)] = data.at[r,str(c)]
    return data_filled


# In[8]:


filled_data = fill_missing(nan_data)
filled_data


# In[9]:


train = filled_data[(filled_data["日期"]>="2020-10-1")&(filled_data["日期"]<="2020-11-30")]
train


# In[10]:


test = filled_data[(filled_data["日期"]>="2020-12-1")&(filled_data["日期"]<="2020-12-31")]
test


# In[11]:


#測試集
merged_test = test[0:18]
merged_test = merged_test[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']]
merged_test.reset_index(drop=True, inplace=True)
for i in range(1,31):
    small_merge = test[i*18:(i+1)*18]
    small_merge = small_merge[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']]
    small_merge.reset_index(drop=True, inplace=True)
    merged_test = pd.concat([merged_test,small_merge],axis = 1,ignore_index=True)
print(merged_test)


# In[12]:


#訓練集建立時間序列
merged_train = train[0:18]
merged_train = merged_train[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']]
merged_train.reset_index(drop=True, inplace=True)
for i in range(1,61):
    small_merge = train[i*18:(i+1)*18]
    small_merge = small_merge[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']]
    small_merge.reset_index(drop=True, inplace=True)
    merged_train = pd.concat([merged_train,small_merge],axis = 1,ignore_index=True)
print(merged_train)


# In[13]:


#pm2.5為第10個column->column[9]
#只取pm2.5特徵
#處理train
training = merged_train[[0,1,2,3,4,5,6]]
training = training.loc[9]
training.reset_index(drop=True, inplace=True)
df_training = training.to_frame().T
for i in range(1,1458):
    train = merged_train[[i,i+1,i+2,i+3,i+4,i+5,i+6]]
    train = train.loc[9]
    train.reset_index(drop=True, inplace=True)
    train = train.to_frame().T
    df_training = pd.concat([df_training,train],ignore_index=True)  
x_train = df_training[[0,1,2,3,4,5]]
y_train = df_training[6]
print(x_train)
print(y_train)


# In[14]:


#處理test
testing = merged_test[[0,1,2,3,4,5,6]]
testing = testing.loc[9]
testing.reset_index(drop=True, inplace=True)
df_testing = testing.to_frame().T
for i in range(1,738):
    test = merged_test[[i,i+1,i+2,i+3,i+4,i+5,i+6]]
    test = test.loc[9]
    test.reset_index(drop=True, inplace=True)
    test = test.to_frame().T
    df_testing = pd.concat([df_testing,test],ignore_index=True)  
x_test = df_testing[[0,1,2,3,4,5]]
y_test = df_testing[6]
print(x_test)
print(y_test)


# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
#regression,預測一小時後，只看pm2.5特徵
mae


# In[16]:


from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(x_train,y_train)
y_pred = xgbc.predict(x_test)
#xgboost,預測一小時後，只看pm2.5
mae = mean_absolute_error(y_test,y_pred)
mae


# In[17]:


#預測六個小時後
#training集
#6為11小時後
training = merged_train[[0,1,2,3,4,5,11]]
training = training.loc[9]
training.reset_index(drop=True, inplace=True)
df_training = training.to_frame().T
for i in range(1,1453):
    train = merged_train[[i,i+1,i+2,i+3,i+4,i+5,i+11]]
    train = train.loc[9]
    train.reset_index(drop=True, inplace=True)
    train = train.to_frame().T
    df_training = pd.concat([df_training,train],ignore_index=True)  
x_train = df_training[[0,1,2,3,4,5]]
y_train = df_training[6]
print(x_train)
print(y_train)


# In[18]:


#處理test
#6為11小時後
testing = merged_test[[0,1,2,3,4,5,11]]
testing = testing.loc[9]
testing.reset_index(drop=True, inplace=True)
df_testing = testing.to_frame().T
for i in range(1,733):
    test = merged_test[[i,i+1,i+2,i+3,i+4,i+5,i+11]]
    test = test.loc[9]
    test.reset_index(drop=True, inplace=True)
    test = test.to_frame().T
    df_testing = pd.concat([df_testing,test],ignore_index=True)  
x_test = df_testing[[0,1,2,3,4,5]]
y_test = df_testing[6]
print(x_test)
print(y_test)


# In[19]:


regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
#regression,預測六小時後，只看pm2.5特徵
mae


# In[20]:


xgbc = XGBClassifier()
xgbc.fit(x_train,y_train)
y_pred = xgbc.predict(x_test)
#xgboost,預測六小時後，只看pm2.5
mae = mean_absolute_error(y_test,y_pred)
mae


# In[21]:


#看全部特徵
#預測一小時後
#處理train
before_training = merged_train[[0,1,2,3,4,5]]
training_first = before_training.loc[0]
for i in range(1,18):
    training = before_training.loc[i]
    training.reset_index(drop=True, inplace=True)
    training_first = pd.concat([training_first,training],ignore_index=True) 
    df_training = training_first.to_frame().T
for k in range(1,1458):
    before_training_other = merged_train[[k,k+1,k+2,k+3,k+4,k+5]]
    training_other = before_training_other.loc[0]
    for i in range(1,18):
        training = before_training_other.loc[i]
        training.reset_index(drop=True, inplace=True)
        training_other = pd.concat([training_other,training],ignore_index=True) 
        train = training_other.to_frame().T
    df_training = pd.concat([df_training,train],ignore_index=True) 
x_train = df_training
#y_train
#pm2.5特徵是9
y_train = merged_train[6]
y_train = y_train.loc[9]
y_train = pd.Series(y_train)
for k in range(1,1458):
    y_training = merged_train[k+6]
    y_training = y_training.loc[9]
    y_training = pd.Series(y_training)
    y_train = pd.concat([y_train,y_training],ignore_index=True)
print(x_train.info())
print(y_train)


# In[22]:


#看全部特徵
#預測一小時後
#處理test
before_testing = merged_test[[0,1,2,3,4,5]]
testing_first = before_testing.loc[0]
for i in range(1,18):
    testing = before_testing.loc[i]
    testing.reset_index(drop=True, inplace=True)
    testing_first = pd.concat([testing_first,testing],ignore_index=True) 
    df_testing = testing_first.to_frame().T
for k in range(1,738):
    before_testing_other = merged_test[[k,k+1,k+2,k+3,k+4,k+5]]
    testing_other = before_testing_other.loc[0]
    for i in range(1,18):
        testing = before_testing_other.loc[i]
        testing.reset_index(drop=True, inplace=True)
        testing_other = pd.concat([testing_other,testing],ignore_index=True) 
        test = testing_other.to_frame().T
    df_testing = pd.concat([df_testing,test],ignore_index=True) 
x_test = df_testing
#y_test
#pm2.5特徵是9
y_test = merged_test[6]
y_test = y_test.loc[9]
y_test = pd.Series(y_test)
for k in range(1,738):
    y_testing = merged_test[k+6]
    y_testing = y_testing.loc[9]
    y_testing = pd.Series(y_testing)
    y_test = pd.concat([y_test,y_testing],ignore_index=True)
print(x_test.info())
print(y_test)


# In[23]:


regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
#regression,預測一小時後，看全部特徵
mae


# In[24]:


xgbc = XGBClassifier()
xgbc.fit(x_train,y_train)
y_pred = xgbc.predict(x_test)
#xgboost,預測一小時後，看全部
mae = mean_absolute_error(y_test,y_pred)
mae


# In[25]:


#看全部特徵
#預測六小時後
#處理train
before_training = merged_train[[0,1,2,3,4,5]]
training_first = before_training.loc[0]
for i in range(1,18):
    training = before_training.loc[i]
    training.reset_index(drop=True, inplace=True)
    training_first = pd.concat([training_first,training],ignore_index=True) 
    df_training = training_first.to_frame().T
for k in range(1,1453):
    before_training_other = merged_train[[k,k+1,k+2,k+3,k+4,k+5]]
    training_other = before_training_other.loc[0]
    for i in range(1,18):
        training = before_training_other.loc[i]
        training.reset_index(drop=True, inplace=True)
        training_other = pd.concat([training_other,training],ignore_index=True) 
        train = training_other.to_frame().T
    df_training = pd.concat([df_training,train],ignore_index=True) 
x_train = df_training
#y_train
#pm2.5特徵是9
y_train = merged_train[11]
y_train = y_train.loc[9]
y_train = pd.Series(y_train)
for k in range(1,1453):
    y_training = merged_train[k+11]
    y_training = y_training.loc[9]
    y_training = pd.Series(y_training)
    y_train = pd.concat([y_train,y_training],ignore_index=True)
print(x_train.info())
print(y_train)


# In[26]:


#看全部特徵
#預測小時後
#處理test
before_testing = merged_test[[0,1,2,3,4,5]]
testing_first = before_testing.loc[0]
for i in range(1,18):
    testing = before_testing.loc[i]
    testing.reset_index(drop=True, inplace=True)
    testing_first = pd.concat([testing_first,testing],ignore_index=True) 
    df_testing = testing_first.to_frame().T
for k in range(1,733):
    before_testing_other = merged_test[[k,k+1,k+2,k+3,k+4,k+5]]
    testing_other = before_testing_other.loc[0]
    for i in range(1,18):
        testing = before_testing_other.loc[i]
        testing.reset_index(drop=True, inplace=True)
        testing_other = pd.concat([testing_other,testing],ignore_index=True) 
        test = testing_other.to_frame().T
    df_testing = pd.concat([df_testing,test],ignore_index=True) 
x_test = df_testing
#y_test
#pm2.5特徵是9
y_test = merged_test[11]
y_test = y_test.loc[9]
y_test = pd.Series(y_test)
for k in range(1,733):
    y_testing = merged_test[k+11]
    y_testing = y_testing.loc[9]
    y_testing = pd.Series(y_testing)
    y_test = pd.concat([y_test,y_testing],ignore_index=True)
print(x_test.info())
print(y_test)


# In[27]:


regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
#regression,預測六小時後，看全部特徵
mae


# In[28]:


xgbc = XGBClassifier()
xgbc.fit(x_train,y_train)
y_pred = xgbc.predict(x_test)
#xgboost,預測一小時後，看全部
mae = mean_absolute_error(y_test,y_pred)
mae


# In[ ]:




