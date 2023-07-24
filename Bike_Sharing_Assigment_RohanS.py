#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("C:/Users/Rohan/Downloads/day.csv")
df.head()


# In[3]:


df.info()


# In[4]:


100*df.isnull().mean()


# In[5]:


df.describe()


# In[6]:


df =df.drop(columns=['instant'])


# In[7]:


df.head()


# In[8]:


df['dteday'] =df['dteday'].apply(lambda x: int(x.split('-')[0]))
df = df.rename(columns={"dteday": "date"})


# In[9]:


df.season.describe()


# In[10]:


df['season'] = df['season'].map({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})


# In[11]:


df['weathersit'] = df['weathersit'].map({1: 'clear', 2: 'mist', 3: 'light', 4: 'heavy'})


# In[12]:


df['weekday']=df['weekday'].map({0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'})


# In[13]:


df['mnth']=df['mnth'].map({1:'jan',2:'feb',3:'march',4:'april',5:'may',6:'june',7:'july',8:'Aug',9:'sept',10:'oct',11:'Nov',12:'Dec'})


# In[14]:


df_num_col=['temp','atemp','hum','windspeed']
plt.figure(figsize=(10, 10))
i=1
for col in df_num_col:
    plt.subplot(2,2,i)
    sns.boxplot(data=df, x=col)
    plt.title(col+" distribution")
    i=i+1
plt.subplots_adjust(left=0.2,bottom=0.5,right=0.9,top=1,wspace=0.1, hspace=0.5)
plt.show()


# In[16]:


sns.boxplot(data=df, x='hum')
plt.title(" humidity distribution")
plt.show()


# In[18]:


df[df['hum'] == 0]


# In[20]:


df[(df.index < (68+10)) & (df.index > (68-10))].hum


# In[22]:


sns.boxplot(data=df, x='hum')
plt.title(" humidity distribution")
plt.show()


# In[23]:


df_num_col=['temp','atemp','hum','windspeed','cnt']


# In[24]:


sns.pairplot(df,vars=df_num_col)
plt.show()


# In[25]:


df_cat_col = ['season','yr','mnth','holiday','weekday','workingday','weathersit']
plt.figure(figsize=(15, 15))
sns.set(style="darkgrid")
i=1
for col in df_cat_col:
    plt.subplot(4,2,i)
    sns.boxplot(data=df, x=col, y='cnt')
    i=i+1
plt.show()


# In[26]:


sns.pairplot(df)
plt.show()


# In[27]:


cor=df.corr()
plt.figure(figsize=(15,5))
sns.heatmap(cor, annot = True)
plt.show()


# In[28]:


category_col = ['mnth', 'season', 'weekday', 'weathersit']
dummy_col = pd.get_dummies(df[category_col], drop_first=True)
dummy_col.head(10)


# In[29]:


df_new = pd.concat([df,dummy_col], axis=1)
df_new = df_new.drop(columns=category_col)


# In[31]:


df_new.info()


# In[34]:


from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_new, test_size=0.3, random_state=100)


# In[35]:


print("shape of traing data", df_train.shape)
print("shape of test data", df_test.shape)


# In[36]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[37]:


num_vars = ['temp','atemp','hum','windspeed','cnt']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[38]:


df_train.describe()


# In[39]:


plt.figure(figsize = (20, 10))
sns.heatmap(round(df_train.corr(),1), annot = True, cmap="YlGnBu")
plt.show()


# In[40]:


y_train = df_train.pop('cnt')
X_train = df_train


# In[42]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)


# In[43]:


from sklearn.feature_selection import RFE
rfe = RFE(lr,n_features_to_select=15)
rfe=rfe.fit(X_train,y_train)


# In[44]:


rfe_df = pd.DataFrame({'feature': X_train.columns, 'Select Status': rfe.support_, 'Ranking': rfe.ranking_})
rfe_df.sort_values(by='Ranking')


# In[45]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
def print_vif(cols):
    df1 = X_train[cols]
    vif_df = pd.DataFrame()
    vif_df['Features'] = df1.columns
    vif_df['VIF'] = [variance_inflation_factor(df1.values, i) for i in range(df1.shape[1])]
    vif_df['VIF'] = round(vif_df['VIF'],2)
#return vif_df
print(vif_df.sort_values(by='VIF',ascending=False))


# In[47]:


import statsmodels.api as sm
col = X_train.columns[rfe.support_]
X_train_rfe = X_train[col]
X_train_sm = sm.add_constant(X_train_rfe)
lr = sm.OLS(y_train, X_train_sm)
lr_model = lr.fit()
print(lr_model.summary())
print("..")
print_vif(col)


# In[49]:


col = col.drop(['workingday'])
X_train_rfe = X_train[col]
X_train_sm = sm.add_constant(X_train_rfe)
lr = sm.OLS(y_train, X_train_sm)
lr_model = lr.fit()
print(lr_model.summary())
print("...")
print_vif(col)


# In[50]:


col = col.drop(['hum'])
X_train_rfe = X_train[col]
X_train_sm = sm.add_constant(X_train_rfe)
lr = sm.OLS(y_train, X_train_sm)
lr_model = lr.fit()
print(lr_model.summary())
print(".")
print_vif(col)


# In[51]:


col = col.drop(['temp'])
X_train_rfe = X_train[col]
X_train_sm = sm.add_constant(X_train_rfe)
lr = sm.OLS(y_train, X_train_sm)
lr_model = lr.fit()
print(lr_model.summary())
print("....")
print_vif(col)


# In[52]:


col = ['yr', 'holiday', 'windspeed',  'mnth_sept', 'season_spring',
       'season_summer', 'season_winter', 'weekday_Sunday', 'weathersit_light',
       'weathersit_mist','temp']
X_train_rfe = X_train[col]
X_train_sm = sm.add_constant(X_train_rfe)
lr = sm.OLS(y_train, X_train_sm)
lr_model = lr.fit()
print(lr_model.summary())
print("**")
print_vif(col)


# In[53]:


y_train_pred = lr_model.predict(X_train_sm)


# In[54]:


sns.distplot(y_train - y_train_pred)
plt.title('Error Terms')                  
plt.xlabel('Errors')  
plt.show()


# In[55]:


residual= y_train_pred - y_train 
sns.regplot(x= y_train_pred, y=residual)
plt.title("Residual and Predicted values")
plt.xlabel("Predicted value")
plt.ylabel("Residual")
plt.show()


# In[56]:


sns.regplot(x= y_train, y=y_train_pred)
plt.title("Residual and Predicted values")
plt.xlabel("y_train")
plt.ylabel("y_train_pred")
plt.show()


# In[57]:


num_vars =['temp','atemp','hum','windspeed','cnt']
df_test[num_vars] = scaler.transform(df_test[num_vars])


# In[58]:


y_test = df_test.pop('cnt')
X_test = df_test


# In[59]:


col = ['yr', 'holiday', 'windspeed',  'mnth_sept', 
       'season_summer', 'season_winter', 'weekday_Sunday', 'weathersit_light',
       'weathersit_mist','temp']
X_test_sm = X_test[col]


# In[ ]:


y_pred = lr_model.predict(X_test_sm)


# In[ ]:




