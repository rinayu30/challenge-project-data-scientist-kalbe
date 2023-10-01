#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ML Regression
#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA


# In[2]:


#import dataset customer
df_customer= pd.read_csv("D:\INTERNSHIP\Kalbe - Rakamin\Week 4\Case Study Data Scientist\Case Study - Customer.csv", sep=';')
df_customer.head()


# In[3]:


#import dataset product
df_product= pd.read_csv("D:\INTERNSHIP\Kalbe - Rakamin\Week 4\Case Study Data Scientist\Case Study - Product.csv", sep=';')
df_product.head()


# In[4]:


#import dataset store
df_store= pd.read_csv("D:\INTERNSHIP\Kalbe - Rakamin\Week 4\Case Study Data Scientist\Case Study - Store.csv", sep=';')
df_store.head()


# In[5]:


#import dataset transaction
df_transaction= pd.read_csv("D:\INTERNSHIP\Kalbe - Rakamin\Week 4\Case Study Data Scientist\Case Study - Transaction.csv", sep=';')
df_transaction.head()


# In[6]:


#Data Cleansing
#Detect the missing value of customer
df_customer.isnull()


# In[7]:


#Detect the missing value of product
df_product.isnull()


# In[8]:


#Detect the missing value of store
df_store.isnull()


# In[9]:


#Detect the missing value of transaction
df_transaction.isnull()


# In[10]:


# Data Cleansing
df_transaction['Date'] = pd.to_datetime(df_transaction['Date'], format='%d/%m/%Y')
# fill missing values
df_customer.isna().sum()
df_customer.fillna(method='ffill', inplace=True)


# In[11]:


# Data Merge
merged_df = pd.merge(df_transaction, df_product, on='ProductID', how='left')
merged_df = pd.merge(merged_df, df_store, on='StoreID', how='left')
merged_df = pd.merge(merged_df, df_customer, on='CustomerID', how='left')
merged_df.head


# In[12]:


merged_df.info()


# In[13]:


#checking
merged_df.duplicated().sum()


# In[14]:


#Regression
df_regression = merged_df.groupby(['Date']).agg({'Qty':
                                                 'sum'}).reset_index()
df_regression


# In[15]:


# Decomposition Analysis (Trend, Seasonal, Residual)
decomposed = seasonal_decompose(df_regression.set_index('Date'))

plt.figure(figsize=(10, 10))

plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')

plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonal')

plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residual')

plt.tight_layout()
plt.show()


# In[16]:


from statsmodels.tsa.stattools import adfuller

# Uji ADF pada kolom 'Qty' dari DataFrame 'df_regression'
result = adfuller(df_regression['Qty'])

# Menampilkan hasil uji ADF
print('Augmented Dickey-Fuller Test Results:')
print('ADF Statistic: %.4f' % result[0])
print('p-value: %.4f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.4f' % (key, value))

# Menyimpulkan hasil uji berdasarkan p-value
if result[1] <= 0.05:
    print('\nHasil uji menunjukkan bahwa data adalah stasioner (Reject H0)')
else:
    print('\nHasil uji menunjukkan bahwa data bukan stasioner (Fail to Reject H0)')


# In[17]:


# split into training and testing data
split_size = round(df_regression.shape[0] * 0.8)

df_train = df_regression.iloc[:split_size]
df_test = df_regression.iloc[split_size:].reset_index(drop=True)

df_train.shape, df_test.shape


# In[18]:


plt.figure(figsize=(10, 5))

sns.lineplot(data=df_train, x=df_train.index, y=df_train['Qty'])
sns.lineplot(data=df_test, x=df_test.index, y=df_test['Qty'])

plt.show()


# In[19]:


from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df_regression['Qty']);


# In[20]:


# import sarimax
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
get_ipython().run_line_magic('matplotlib', 'inline')
# grid search for p, d, and q
auto_arima_model = pm.auto_arima(
    df_train['Qty'], 
    seasonal=False, 
    stepwise=False, 
    suppress_warnings=True, 
    trace = True
)
auto_arima_model.summary()


# In[21]:


#import sarimax
p, d, q = auto_arima_model.order
model = SARIMAX(df_train['Qty'].values, order=(p,d,q))
model_fit = model.fit(disp=False)


# In[22]:


# Set the index for df_train and df_test
df_train = df_train.set_index(['Date'])
df_test = df_test.set_index(['Date'])
#count rsme
from sklearn.metrics import mean_squared_error, mean_absolute_error
predictions = model_fit.predict(start=len(df_train), end=len(df_train)+len(df_test)-1)
rmse = mean_squared_error(df_test, predictions, squared=False)
rmse


# In[23]:


# forecasting for next 90 days
period = 90
forecast = model_fit.forecast(steps=period)
index = pd.date_range(start='01-01-2023', periods=period)
df_forecast = pd.DataFrame(forecast, index=index, columns=['Qty'])


# In[24]:


plt.figure(figsize=(12,8))
plt.title('Forecasting Sales')
plt.plot(df_train, label='Train')
plt.plot(df_test, label='Test')
plt.plot(df_forecast, label='Predicted')
plt.legend(loc='best')
plt.show()


# In[25]:


# plot forecast
df_forecast.plot(figsize=(12,8), title='Forecasting Sales', xlabel='Date', ylabel='Total Qty', legend=False)


# In[26]:


# forecast product for next 90 days
warnings.filterwarnings('ignore')

product_reg_df = merged_df[['Qty', 'Date', 'Product Name']]
new = product_reg_df.groupby("Product Name")

forecast_product_df = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=90)})

for product_name, group_data in new:
    target_var = group_data['Qty']
    model = SARIMAX(target_var.values, order=(p,d,q))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(90)
    forecast_product_df[product_name] = forecast

forecast_product_df.set_index('Date', inplace=True)
forecast_product_df


# In[27]:


# plot forecast for products
plt.figure(figsize=(12,8))
for i in forecast_product_df.columns:
    plt.plot(forecast_product_df[i], label=i)
plt.legend(loc=6, bbox_to_anchor=(1,.82))
plt.title('Forecasting Product')
plt.xlabel('Date')
plt.ylabel('Total Qty')
plt.show()


# In[28]:


#CLUSTERING
# Membuat data baru untuk clustering, yaitu groupby by customerID lalu yang di aggregasi
df_cluster = merged_df.groupby('CustomerID').agg({'TransactionID': 'count','Qty': 'sum','TotalAmount': 'sum'}).reset_index()
df_cluster


# In[29]:


#Normalize
from sklearn.preprocessing import normalize

data_cluster = df_cluster.drop(columns=['CustomerID'])
data_cluster_normalize = normalize(data_cluster)
data_cluster_normalize


# In[30]:


#use Kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

K = range(2, 8)
fits = []
score = []

for k in K:
    model = KMeans(n_clusters=k, random_state=0, n_init=1).fit(data_cluster_normalize)
    fits.append(model)
    score.append(silhouette_score(data_cluster_normalize, model.labels_, metric='euclidean'))


# In[31]:


# Visualisasi silhouette score 
sns.lineplot(x = K, y = score);


# In[32]:


fits[2]
print(fits[2])

df_cluster['cluster label'] = fits[2].labels_

# Mengelompokkan DataFrame
df_cluster.groupby(['cluster label']).agg({'CustomerID':'count',
                                           'TransactionID' : 'mean', 
                                           'Qty': 'mean',
                                           'TotalAmount' : 'mean'})


# In[34]:


#Visualisasi clustering
plt.figure(figsize=(3,3))
sns.pairplot(data=df_cluster,hue='cluster label',palette='Set2')
plt.show()


# In[ ]:




