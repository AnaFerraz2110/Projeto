#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install yfinance --upgrade --no-cache-dir')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Importando Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import yfinance as yf

#Importando Bibliotecas de Aprendizado de Máquina
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[3]:


yf.pdr_override()


# In[4]:


# Defina os ativos que deseja baixar os dados
ticker = ['^BVSP']

# Defina o intervalo de tempo para o qual deseja baixar os dados
start_date = "2015-01-01"
end_date = "2020-06-01"

# Tente baixar os dados usando o pandas-datareader
df_ibov= web.get_data_yahoo(ticker,start=start_date, end=end_date)

X = df_ibov[['Open']]
y = df_ibov[['Close']]


# In[5]:


df_ibov.head(10)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)


# In[7]:


print (y_train)


# In[8]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[9]:


y_pred = regressor.predict(X_test)


# In[10]:


print('intercept:', regressor.intercept_)
print('Coeficient:', regressor.coef_)


# In[11]:


df_result = pd.DataFrame
df_result = X_test.copy()
df_result['Close'] = y_test.copy()
df_result['Prediction'] = y_pred

df_result.head()


# In[12]:


plt.style.use('default')
df_bar = df_result[['Close', 'Prediction']].head(10)

df_bar


# In[13]:


df_bar = df_result[['Close', 'Prediction']].head(30)
df_bar.plot(kind='bar',figsize=(10,5))
plt.title('Amostras da Predição vs. Real ')
plt.show()


# In[14]:


df_bar.plot.line(figsize=(16, 5))
plt.title('Predição do Índice IBOVESPA vs. Real ')
plt.xlabel('Dias')
plt.ylabel('Pontos IBOV')
plt.legend(['Close', 'Prediction'])
plt.show()


# In[20]:


plt.style.use('default')
plt.scatter(X_train, y_train, color='orange')
plt.plot(regressor.predict(X_train),X_train['Open'], color='blue')
plt.title('Abertura vs. Fechamento ')
plt.xlabel('Abertura')
plt.ylabel('Fechamento')
plt.show()


# In[21]:


# Base de Testes
# Comparar o fechamento real com o fechamento predito pelo modelo (Base de Teste 80%)
plt.style.use('default')
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test['Open'], y_pred, color='orange')
plt.title('Abertura vs. Fechamento (Base de Testes)')
plt.xlabel('Abertura')
plt.ylabel('Fechamento')
plt.show()


# In[17]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# In[18]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  # em média, meu modelo erra 746 pontos do ibov
print('Mean Absolute Percentage Error', mean_absolute_percentage_error(y_test, y_pred)) # o erro percenteu para todo o modelo é de 1%
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[19]:


print(regressor.score(X_test, y_test))


# In[ ]:




