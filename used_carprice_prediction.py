#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df = pd.read_csv("cars_price.csv")


# In[3]:


df.head()


# In[4]:


df.isna().sum()


# In[5]:


df['volume(cm3)'].fillna(df['volume(cm3)'].median(),inplace=True)


# In[6]:


df.drop(['segment'],axis=1,inplace=True)


# In[7]:


# df.dropna(inplace=True,how='any',axis=0)
df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()


# In[8]:


df.isna().sum()


# In[9]:


df.shape


# In[10]:


# def new_col(x):
#     y = 2020-x
#     return (x+y)


# In[11]:


# df['newcol']=df['year'].map(lambda x:new_col(x))


# In[12]:


# df['year']=pd.to_datetime(df['year'])
# df['newcol']=pd.to_datetime(df['newcol'])
# df["year"]=(df["newcol"]-df['year']).dt.total_seconds() 


# In[13]:


df.head()


# # Exploratory Data Analysis
# 

# In[14]:


len(df.make.unique())


# In[15]:


plt.figure(figsize=(15,8))
sns.countplot(data = df, x="color",order = df['color'].value_counts().index)


#  black,silver and blue color cars are mostly bought compared to other colors

# In[16]:


plt.subplot(1,2,1)
plt1 = df.fuel_type.value_counts().plot('bar',color=['C1','C2'])
plt.title("fuel_typecountplot")
plt1.set(xlabel="fuel_type")
plt.subplot(1,2,2)
plt1 = df.transmission.value_counts().plot('bar',color=['C3','C4'])
plt.title("transmission_count")
plt1.set(xlabel='transmission_type',ylabel='count')


# petrol type are most prefered than disel and electrocars are rarely bought
# 

# In[17]:


sns.countplot(data = df, x="drive_unit",order = df['drive_unit'].value_counts().index)


# In[18]:


sns.countplot(x='fuel_type', hue='condition', data=df)


#             petrol and disel with_milage are sold more

# In[19]:


df.make.value_counts()[:15].sum()


# In[20]:


maps=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10']


# In[21]:


plt.figure(figsize=(15,8))
plt.xlabel("make")
plt.ylabel("count")
df.make.value_counts()[:15].plot(kind='bar',color=maps)


# these are top 15 most purchased car companies 
# volkswagan is most_purchased car following audi,bmw

# In[22]:


plt.figure(figsize=(15,8))
plt.ylabel("total price(USD) sum")
df.groupby('make')['priceUSD'].sum().sort_values(ascending=False)[:15].plot('bar',rot=80,color=maps)


# In[23]:


plt.figure(figsize=(15,8))
plt.ylabel("total price(USD) sum")
df.groupby('make')['priceUSD'].median().sort_values(ascending=False)[:15].plot('bar',colormap='inferno',legend=True)

mclaren,bentley are most costly cars and follwed by aston-martin and tesla
# In[24]:


plt.figure(figsize=(15,8))
df.groupby(['make','fuel_type'])['priceUSD'].count().sort_values(ascending=False)[:15].plot('bar',colormap='prism',rot=80)

from the above graph it is evident that most of the cars runs on petrol volswagan and bmw are the only cars
in top 15 which uses both petrol and diesel
# In[ ]:





# In[25]:


df.groupby('make')['priceUSD'].median()


# In[ ]:





# In[26]:



# make_median =dict(df.groupby('make')['priceUSD'].median())
# df['make'].replace(make_median ,inplace=True)


# In[27]:


# plt.scatter(df['make'],df['priceUSD'])


# In[28]:


# df['make'] = df['make'].map(lambda x: (math.log(x)))


# In[29]:


# plt.scatter(df['make'],df['priceUSD'])


# In[30]:


plt.scatter(df['year'],df['priceUSD'])
plt.xlabel("year")
plt.ylabel('price')


# In[31]:


# df['volume(cm3)'] = df['volume(cm3)'].map(lambda x: (math.log(x)))


# In[32]:


plt.scatter(df['volume(cm3)'],df['priceUSD'])


# In[ ]:





# In[33]:


# labelencoding based on the count
def label_encoding(column):
    cat_feature_value_counts = column.value_counts()
    value_counts_list = cat_feature_value_counts.index.tolist()
    value_counts_range = list(reversed(range(1,len(cat_feature_value_counts)+1)))
    encoder_list_range = dict(zip(value_counts_list,value_counts_range))
    column = column.map(encoder_list_range)
    return column
    


# In[34]:


df['color'] = label_encoding(df['color'])
df['make'] = label_encoding(df['make'])
df['drive_unit']=label_encoding(df['drive_unit'])
df['condition']=label_encoding(df['condition'])


# In[35]:


df.drop(['Unnamed: 0','model'],axis=1,inplace=True)


# In[36]:


df =pd.get_dummies(df,drop_first=True)


# In[37]:


#calculating corelation
core = df.corr().abs()
core


# In[38]:


plt.figure(figsize=(25,25))
sns.heatmap(core,cmap="YlGnBu", annot=True)
plt.show()


# # handling outliers

# In[39]:


df.plot(kind='box')
plt.show()


# In[40]:


sns.boxplot(x='make',data=df)


# In[41]:


df.describe()


# In[42]:


from scipy import stats
z = np.abs(stats.zscore(df))


# In[43]:


df.shape


# In[44]:


print(np.where(z > 3))


# In[45]:


df = df[(z <= 3).all(axis=1)]
df.shape


# In[46]:


df.head()


# In[47]:


# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


# # training and testing different models 

# In[48]:


cols = df.columns
scaler = StandardScaler()
df =pd.DataFrame(scaler.fit_transform(df),columns=cols)
df.head()


# In[49]:


x = df.drop(['priceUSD'],axis=1)
y = df['priceUSD']


# In[50]:


x.shape


# In[51]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# In[52]:


from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


# In[53]:


#defining function to get r2score
def r2score(model,X,y):
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    r2scores = cross_val_score(model,x_train,y_train,cv = cv,scoring = 'r2')
    return(r2scores.mean())
    


# In[54]:


#rootmeansquareerror
def rmse(model,x,y):
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    rmse = -cross_val_score(model,x_train,y_train,cv = cv,scoring = 'neg_mean_squared_error')
    return(np.sqrt(rmse.mean()))


# # linear regression

# In[55]:


try:
    lr = LinearRegression()
    r2_lr=r2score(lr,x_train,y_train)
    rmse_lr = rmse(lr,x_train,y_train)
except:
    r2_lr=r2score(lr,x_train,y_train)
    rmse_lr = rmse(lr,x_train,y_train)


# In[56]:


print("the r2_score on cross_validation",r2_lr)
print("the rmse_score on cross_validation",rmse_lr)


# In[57]:


lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print("the r2 on test",r2_score(y_test,y_pred))
print("the rmse on teat",np.sqrt(mean_squared_error(y_test,y_pred)))


# # support vector regressor

# In[58]:


svr = SVR( gamma='auto')
r2_svr=r2score(svr,x_train,y_train)
rmse_svr = rmse(svr,x_train,y_train)


# In[59]:


print("the r2_score on cross_validation",r2_svr)
print("the rmse_score on cross_validation",rmse_svr)


# In[60]:


# param_grid = {'C':[0.001,0.1,1,10],'gamma':[1,0.1,0.01,0.001] ,'kernel':['rbf']}
# from sklearn.model_selection import RandomizedSearchCV
# clf = RandomizedSearchCV(svr,param_grid,n_jobs=-1,verbose=4,cv=3,scoring = 'r2')


# In[61]:


# clf.fit(x_train,y_train)


# In[62]:


svr.fit(x_train,y_train)
y_pred = svr.predict(x_test)
print("the r2 on test",r2_score(y_test,y_pred))
print("the rmse on test",np.sqrt(mean_squared_error(y_test,y_pred)))


# # decesion tree regressor

# In[63]:


dtr = DecisionTreeRegressor(random_state=100)
r2_dtr=r2score(dtr,x_train,y_train)
rmse_dtr = rmse(dtr,x_train,y_train)
print("the r2_score on cross_validation",r2_dtr)
print("the rmse_score on cross_validation",rmse_dtr)


# In[64]:


dtr.fit(x_train,y_train)
y_pred = dtr.predict(x_test)
print("the r2 on test",r2_score(y_test,y_pred))
print("the rmse on test",np.sqrt(mean_squared_error(y_test,y_pred)))


# # random forest regressor

# In[65]:


rfr = RandomForestRegressor(random_state=42,bootstrap=True,n_estimators=50,max_features='log2')
r2_rfr=r2score(rfr,x_train,y_train)
rmse_rfr = rmse(rfr,x_train,y_train)
print("the r2_score on cross_validation",r2_rfr)
print("the rmse_score on cross_validation",rmse_rfr)


# In[66]:


rfr.fit(x_train,y_train)
y_pred = rfr.predict(x_test)
print("the r2 on test",r2_score(y_test,y_pred))
print("the rmse on test",np.sqrt(mean_squared_error(y_test,y_pred)))

