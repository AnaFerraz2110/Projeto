#!/usr/bin/env python
# coding: utf-8

# In[2]:


#instalar o Yahoo Finance e o Pandas DataReader
get_ipython().system('pip install yfinance')
get_ipython().system('pip install pandas-datareader')
get_ipython().system('pip install --upgrade numpy')

#Importando Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import yfinance as yf

#Importando Bibliotecas de Aprendizado de Máquina
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#configurar opções de visualização
sns.set(style='darkgrid', context='talk', palette='Dark2')
pd.options.display.max_columns = None
pd.options.display.max_rows = None

#desativar avisos
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Permite que o pandas-datareader obtenha dados do Yahoo retorna um DataFrame
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


df_ibov.head(10)


# In[6]:


df_ibov.describe()


# In[7]:


#verificar se há valores nulos
df_ibov.isna().sum()


# In[8]:


# Variavel preço Fechado
df_close = df_ibov['Close']
# converter Dataframe Panda
df_close = df_close.to_frame()


# In[9]:


plt.figure(figsize=(10, 4))
plt.title('IBOV')
plt.xlabel('Days')
plt.ylabel('IBOV Points')
plt.plot(df_close['Close'], '-o', linewidth=2, markersize=3) # add linewidth and markersize to the close price line
plt.plot(df_close.rolling(window=30).mean()['Close'], '-s', linewidth=2, markersize=3) # add linewidth and markersize to the mean line
plt.legend(['Close Price', 'Mean'])
plt.grid(True) # add grid
plt.show()


# In[10]:


plt.figure(figsize=(10, 4))
plt.title('IBOV')
plt.xlabel('Days')
plt.ylabel('IBOV Points')
plt.plot(df_close['Close'])
plt.plot(df_close.rolling(window=90).mean()['Close'])  # mean last 30 days
plt.legend(['Close Price', 'Mean'])
plt.show()


# In[11]:


# Criando um DataFrame vazio
df_acoes = pd.DataFrame()

# Lista de Tickers
tickers = ['ITUB4.SA', 'MGLU3.SA','^BVSP']

# Loop para obter os dados de cada ticker
for ticker in tickers:
    df_acoes[ticker] = yf.download(ticker, start=start_date, end=end_date)['Close']

# Renomeando as colunas
df_acoes.rename(columns = {'ITUB4.SA': 'ITUB4', 'MGLU3.SA':'MGLU3', '^BVSP':'IBOV'}, inplace=True)

# Normalizando os dados do IBOV (dividindo por 1000)
df_acoes['IBOV'] /= 1000


# In[12]:


#redefinindo o índice (usando apenas de 0 até o comprimento)
df_acoes.reset_index(inplace=True)

#removendo valores ausentes do dataframe
df_acoes.dropna(inplace=True)

#verificando se há valores ausentes restantes
print("Valores ausentes: ", df_acoes.isna().sum())

#confirmando o dataframe
print(df_acoes.head())


# In[13]:


print(df_acoes.isna().sum())


# In[14]:


tickers = list(df_acoes.drop(["Date"], axis = 1).columns)
print(tickers)


# In[15]:


plt.figure(figsize=(10, 4))
for ticker in tickers:
  plt.plot(df_acoes['Date'], df_acoes[ticker])

plt.legend(tickers, loc='upper left')
plt.title('Cotação ao longo do Tempo', fontsize = 13)
plt.show()


# In[16]:


plt.figure(figsize=(10,5))
plt.plot(df_acoes['Date'], df_acoes['IBOV']*1000, alpha = 0.8)
plt.plot(df_acoes['Date'],df_acoes['IBOV'].rolling(window = 30).mean()*1000)
plt.plot(df_acoes['Date'],df_acoes['IBOV'].rolling(window = 90).mean()*1000)
plt.plot(df_acoes['Date'],df_acoes['IBOV'].rolling(window = 365).mean()*1000)
plt.title('Cotações diárias e médias móveis do IBOV', fontsize = 15)
plt.legend(['Cotação diária','Média móvel mensal','Média móvel trimestral','Média móvel anual'])
plt.show()


# In[17]:


returns = pd.DataFrame()
for ticker in tickers:
    returns[ticker] = df_acoes[ticker].pct_change() 
returns['Date'] = df_acoes['Date']
returns.describe()


# In[18]:


# distribuição normal
for ticker in tickers:
  sns.distplot(returns[ticker].dropna())

plt.legend(tickers)
plt.show()


# In[19]:


df_ibov_ml = df_ibov


# In[20]:


# variações
import matplotlib.dates as mdates
import datetime as dt

df_ibov_ml['Variation'] = df_ibov_ml['Close'].sub(df_ibov_ml['Open'])

x = df_ibov_ml.index
y = df_ibov_ml ['Variation']

plt.plot_date(x,y, color='r',fmt="r-")
plt.xticks(rotation=30)

plt.show()


# In[21]:


#Importação do Dataset COVID-19 do Kaggle
import pandas as pd
from datetime import datetime
df_pandemia = pd.read_excel("C:/Users/Ana Carolina Ferraz/Documents/PUC_MINAS/TCC/DADOS_PANDEMIA_BRASIL.xlsx")


# In[22]:


df_pandemia.head(10)


# In[23]:


df_covid = df_pandemia.filter(['data', 'regiao', 'casosAcumulado'], axis=1)


# In[24]:


df_covid = df_covid.loc[df_covid['regiao'] == 'Brasil']
df_covid = df_covid.drop('regiao', 1)


# In[25]:


df_covid.reset_index(inplace=True)
df_covid.rename(columns = {'data': 'Date', 'casosAcumulado':'Confirmed Cases'}, inplace=True)


# In[26]:


df_merge = df_acoes.merge(df_covid, on='Date', how='left')
                          
df_merge['Confirmed Cases'] = df_merge['Confirmed Cases'].fillna(0)
print(df_merge)
                          
df_merge['Confirmed Cases'] = df_merge['Confirmed Cases']
df_merge.drop('index', 1, inplace=True)  

                          
                    


# In[27]:


plt.figure(figsize=(16, 5))
for ticker in tickers:
  plt.plot(df_merge['Date'], df_merge[ticker])

plt.legend(tickers)
plt.title('Cotação ao longo do Tempo', fontsize = 13)
plt.show()


# In[28]:


start_filter = '2020-01-01'
end_filter = '2020-06-01'
filter_2020 = (df_merge['Date']>=start_filter) & (df_merge['Date']<=end_filter)
df_merge_2020 = df_merge[filter_2020]

covid_date = df_merge_2020['Date']
covid_cases = df_merge_2020['Confirmed Cases']

plt.figure(figsize=(10, 5))
plt.plot(df_merge_2020['Date'], df_merge_2020['Confirmed Cases'])

plt.title('Casos de Corona Vírus no Brasil', fontsize = 13)
plt.show()


# In[29]:


start_filter = '2020-02-26'
end_filter = '2020-04-30'
filter_2020 = (df_merge['Date']>=start_filter) & (df_merge['Date']<=end_filter)
df_merge_2020 = df_merge[filter_2020]

covid_date = df_merge_2020['Date']
covid_cases = df_merge_2020['Confirmed Cases']

plt.figure(figsize=(15, 5))
plt.plot(df_merge_2020['Date'], df_merge_2020['Confirmed Cases'])

plt.title('Casos de Corona Vírus no Brasil')
plt.show()


# In[30]:


#Análise dos Datasets em Conjunto e Correlação
fig, ax1 = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(10)

x = df_merge['Date']

y2 = df_merge['Confirmed Cases']
for ticker in tickers:
  y1 = df_merge[ticker]
  ax1.plot(x, y1)
    
plt.legend(tickers)
ax2 = ax1.twinx()
ax2.plot(x, y2, 'black')

ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Price', color='g')
ax2.set_ylabel('COVID-19 Confirmed Cases', color='black')


plt.title('Bolsa de Valores x Casos Confirmados de COVID-19')
plt.show()


# In[31]:


#Apenas 2020
inicio_filtro = '2020-01-01'
fim_filtro = '2020-06-01'
filtro_2020 = (df_merge['Date'] >= inicio_filtro) & (df_merge['Date'] <= fim_filtro)
df_merge_2020 = df_merge[filtro_2020]
print(df_merge_2020)

#Criação do Gráfico
fig, eixo1 = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(10)

#Definindo o eixo x
x = df_merge_2020['Date']

#Definindo o eixo y2
y2 = df_merge_2020['Confirmed Cases']

#Definindo o eixo y1
for ticker in tickers:
    y1 = df_merge_2020[ticker]
eixo1.plot(x, y1)

#Definindo a legenda
plt.legend(tickers, loc='upper left')

eixo2 = eixo1.twinx()
eixo2.plot(x, y2, 'red')

eixo1.set_xlabel('Date')
eixo1.set_ylabel('Preço das Ações', color='g')
eixo2.set_ylabel('Casos Confirmados de COVID-19', color='red')

plt.title('Bolsa de Valores x Casos Confirmados de COVID-19')
plt.show()


# In[32]:


start_filter = '2020-01-01'
end_filter = '2020-06-01'
filter_2020 = (df_merge['Date']>=start_filter) & (df_merge['Date']<=end_filter)
df_merge_2020 = df_merge[filter_2020]
df_merge_2020['IBOV'].corr(df_merge_2020['Confirmed Cases'])


# In[33]:


df_merge_2020.corr()


# In[34]:


df_merge_2020['IBOV'].corr(df_merge_2020['ITUB4'])


# In[35]:


import seaborn as sns
sns.heatmap(df_merge_2020.drop('Date', 1).corr(), annot = True)
plt.show()


# In[36]:


df_merge_2020.corr()


# In[37]:


figure = plt.figure(figsize=(9,3))
plt.scatter(np.log(df_merge_2020['IBOV']), np.log(df_merge_2020['Confirmed Cases']))
plt.xlabel('IBOV')
plt.ylabel('Confirmed Cases')


# In[38]:


figure = plt.figure(figsize=(9,3))
plt.scatter((df_merge_2020['IBOV']), (df_merge_2020['ITUB4']))
plt.xlabel('IBOV')
plt.ylabel('ITUB4')


# In[39]:


df_desc = df_merge_2020.set_index('Date').drop('ITUB4', 1).drop('MGLU3', 1)
df_desc['IBOV'] = df_desc['IBOV']*1000
print('Máxima no período')
print(df_desc.max())

print('\nDatas de referência')


# In[ ]:




