#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install yfinance --upgrade --no-cache-dir')


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# importing libraries to the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import yfinance as yf

# importing machine learning libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

plt.style.use('default')


# In[2]:


yf.pdr_override()


# In[3]:


# Defina os ativos que deseja baixar os dados
ticker = ['^BVSP']

# Defina o intervalo de tempo para o qual deseja baixar os dados
start_date = "2015-01-01"
end_date = "2020-06-01"

# Tente baixar os dados usando o pandas-datareader
df_ibov= web.get_data_yahoo(ticker,start=start_date, end=end_date)


# In[4]:


pd.options.display.float_format = '{:.5f}'.format
df_ibov = df_ibov[['Open', 'Close']]
X = df_ibov[['Open']]
y = df_ibov[['Close']]


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[6]:


from sklearn.svm import SVR
regressor_rbf = SVR(kernel = 'rbf')
regressor_rbf.fit(X_train, y_train)

regressor_linear = SVR(kernel = 'linear')
regressor_linear.fit(X_train, y_train)
     


# In[7]:


y_pred_rbf = regressor_rbf.predict(X_test)
y_pred_linear = regressor_linear.predict(X_test)


# In[12]:


plt.style.use('default')
plt.scatter(X_train, y_train, color='orange')
plt.plot(X_train['Open'], regressor_rbf.predict(X_train), color='red', label = 'RBF Model')
plt.plot(X_train['Open'], regressor_linear.predict(X_train), color='blue', label = 'RBF Model')
plt.title('Abertura vs. Fechamento - Treino (SVR)')
plt.xlabel('Abertura')
plt.ylabel('Fechamento')
plt.show()


# In[13]:


plt.scatter(X_test, y_test, color = 'orange', label = 'Orignial Dataset')
plt.plot(X_test['Open'], y_pred_rbf, color = 'red', label = 'RBF Model')
plt.plot(X_test['Open'], y_pred_linear, color = 'blue', label = 'Linear Model')
plt.title('Abertura vs. Fechamento - Teste (SVR)')
plt.xlabel('Variaveis')
plt.ylabel('Fechamento')
plt.legend()
plt.show()


# In[14]:


X2 = df_ibov.iloc[:, 0].values
y2 = df_ibov.iloc[:, -1].values


# In[15]:


df_result = df_ibov.iloc[:, -1]
df_result = df_result.to_frame()
df_result['RBF Model'] = regressor_rbf.predict(X).copy()
df_result['Linear Model'] = regressor_linear.predict(X).copy()
df_result


# In[19]:


df_bar = df_result[['Close', 'RBF Model']].head(30)
df_bar.plot(kind='bar',figsize=(10,5))
plt.title('Amostras Aleatórias da Predição vs. Real (SVR RBF)')
plt.show()


# In[18]:


df_bar = df_result[['Close', 'Linear Model']].head(30)
df_bar.plot(kind='bar',figsize=(10,5))
plt.title('Amostras Aleatórias da Predição vs. Real (SVR Linear)')
plt.show()


# In[20]:


df_result.plot.line(figsize=(16, 5))
plt.title('IBOV')
plt.xlabel('Days')
plt.ylabel('IBOV Points')
plt.legend(['Close', 'RBF Model', 'Linear Model'])
plt.title('Predição do Índice IBOVESPA vs. Real (SVR)')
plt.show()


# In[21]:


print(regressor_rbf.score(X, y))
print(regressor_linear.score(X, y))


# In[ ]:




