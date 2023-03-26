#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install yfinance --upgrade --no-cache-dir')
get_ipython().system('pip install --upgrade numpy')
get_ipython().system('pip install graphviz')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# importing libraries to the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import yfinance as yf

# importing machine learning libraries
from sklearn.tree import DecisionTreeRegressor
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


# In[5]:


pd.options.display.float_format = '{:.5f}'.format


# In[6]:


df_ibov = df_ibov[['Open', 'Close']]


# In[7]:


X = df_ibov[['Open']]
y = df_ibov[['Close']]


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[9]:


regressor = DecisionTreeRegressor(max_depth=5, max_leaf_nodes=5, random_state = 0)
regressor.fit(X_train, y_train)


# In[10]:


y_pred = regressor.predict(X_test)


# In[11]:


import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(regressor, out_file='tree.dot')


# In[12]:


df_result = pd.DataFrame
df_result = X_test.copy()
df_result['Close'] = y_test.copy()
df_result['Prediction'] = y_pred
df_result.head()


# In[13]:


df_bar = df_result[['Close', 'Prediction']].head(30)
df_bar.plot(kind='bar',figsize=(10,5))
plt.title('Amostras Aleatórias da Predição vs. Real (Decision Tree)')
plt.show()


# In[14]:


X_grid = np.arange(min(X_test['Open']), max(X_test['Open']), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_pred, color = 'orange')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Árvore de Decisão')
plt.xlabel('Close')
plt.ylabel('Open')
plt.show()


# In[17]:


plt.style.use('default')
plt.scatter(X_train, y_train, color='orange')
plt.plot(X_train['Open'],regressor.predict(X_train),color='blue')
plt.title('Abertura vs. Fechamento (Base de Treinamento)')
plt.xlabel('Abertura')
plt.ylabel('Fechamento')
plt.show()


# In[60]:


df_result.drop('Open', 1).plot.line(figsize=(10, 5))
plt.title('Predição do Índice IBOVESPA vs. Real')
plt.xlabel('Dias')
plt.ylabel('Pontos IBOV')
plt.legend(['Close', 'Prediction'])
plt.show()


# In[61]:


print(regressor.score(X_test, y_test))


# In[62]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  # em média, meu modelo erra 746 pontos do ibov
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

