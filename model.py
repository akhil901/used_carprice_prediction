#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_csv("cars_price.csv")


df['volume(cm3)'].fillna(df['volume(cm3)'].median(),inplace=True)



df.drop(['segment'],axis=1,inplace=True)

df.dropna(inplace=True,how='any',axis=0)

df.isna().sum()

# labelencoding based on the count
def label_encoding(column):
    cat_feature_value_counts = column.value_counts()
    value_counts_list = cat_feature_value_counts.index.tolist()
    value_counts_range = list(reversed(range(1,len(cat_feature_value_counts)+1)))
    encoder_list_range = dict(zip(value_counts_list,value_counts_range))
    column = column.map(encoder_list_range)
    return column

df['color'] = label_encoding(df['color'])
df['make'] = label_encoding(df['make'])
df['drive_unit']=label_encoding(df['drive_unit'])
df['condition']=label_encoding(df['condition'])

df.drop(['Unnamed: 0','model'],axis=1,inplace=True)

df =pd.get_dummies(df,drop_first=True)

from scipy import stats
z = np.abs(stats.zscore(df))
df = df[(z <= 3).all(axis=1)]

cols = df.columns
scaler = StandardScaler()
df =pd.DataFrame(scaler.fit_transform(df),columns=cols)
df.head()

x = df.drop(['priceUSD'],axis=1)
y = df['priceUSD']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

#defining function to get r2score
def r2score(model,X,y):
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    r2scores = cross_val_score(model,x_train,y_train,cv = cv,scoring = 'r2')
    return(r2scores.mean())

#rootmeansquareerror
def rmse(model,x,y):
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    rmse = -cross_val_score(model,x_train,y_train,cv = cv,scoring = 'neg_mean_squared_error')
    return(np.sqrt(rmse.mean()))


# # Random Forest Regressor


rfr = RandomForestRegressor(random_state=42,bootstrap=True,n_estimators=50,max_features='log2')
r2_rfr=r2score(rfr,x_train,y_train)
rmse_rfr = rmse(rfr,x_train,y_train)
print("the r2_score on cross_validation_data",r2_rfr)
print("the rmse_score on cross_validation_data",rmse_rfr)


rfr.fit(x_train,y_train)
y_pred = rfr.predict(x_test)
print("the r2_score on test_data",r2_score(y_test,y_pred))
print("the rmse_score on test_data",np.sqrt(mean_squared_error(y_test,y_pred)))





