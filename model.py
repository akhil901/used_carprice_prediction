#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#input the data_file you want to predict
df = pd.read_csv("cars_price.csv")


df['volume(cm3)'].fillna(df['volume(cm3)'].median(),inplace=True)

df.drop(['segment'],axis=1,inplace=True)
df.dropna(inplace=True,how='any',axis=0)

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


# In[2]:


import pickle
# Load the Model back from file
Pkl_Filename = "Pickle_RFR_Model.pkl" 
with open(Pkl_Filename, 'rb') as file:  
    Pickled_RFR_Model = pickle.load(file)


# In[3]:


y_pred = Pickled_RFR_Model.predict(x)


# In[5]:


print("the r2_score on test_data",r2_score(y,y_pred))
print("the rmse_score on test_data",np.sqrt(mean_squared_error(y,y_pred)))

