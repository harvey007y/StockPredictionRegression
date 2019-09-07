import pandas as pd
import datetime
#import pandas_datareader.data as web
from pandas import Series, DataFrame
import math
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



start = datetime.datetime(2012, 9, 1)
end = datetime.datetime(2019, 9, 1)

#df = web.DataReader("AAPL", 'yahoo', start, end)
#print(df.tail())

df = pd.read_csv('aapl.csv',encoding='latin-1')
print(df.tail())
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()

rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')

plt.show()

dfreg = df.loc[:, ['Adj Close', 'Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
print(dfreg.head())
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], axis = 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_len = len(X_train)
test_len = len(X_test)
print('train length', len(X_train))
print('test length', len(X_test))


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test, y_test)
confidencepoly3 = clfpoly3.score(X_test, y_test)
confidenceknn = clfknn.score(X_test, y_test)

# results
print('The linear regression confidence is ', confidencereg)
print('The quadratic regression 2 confidence is ', confidencepoly2)
print('The quadratic regression 3 confidence is ', confidencepoly3)
print('The knn regression confidence is ', confidenceknn)

clfreg_preds = clfreg.predict(X_test)

print(clfreg_preds[-5:])
print(y[-5:])

# Plot outputs
#plt.scatter(X_test, y_test,  color='black')
#plt.scatter(y_test, clfreg_preds, color='blue')
datelist = pd.date_range(pd.datetime(2012, 9, 1), periods= train_len + test_len)
datelist_forecast = datelist[-test_len:]
print(datelist_forecast[0])
print(datelist_forecast[test_len - 1])
df = pd.DataFrame(np.cumsum(np.random.randn(train_len + test_len)), 
                  columns=['price'], index=datelist)
df_forecast = pd.DataFrame(np.cumsum(np.random.randn(test_len)), 
                  columns=['price'], index=datelist_forecast)

myctr = 0
for i, row in df.iterrows(): 
  df.at[i,'price'] = y[myctr]
  myctr = myctr + 1
myctr = 0
for i, row in df_forecast.iterrows(): 
  df_forecast.at[i,'price'] = clfreg_preds[myctr]
  myctr = myctr + 1    
plt.figure(figsize=(20,10))
plt.title('Linear Regression')
plt.plot(df_forecast[:test_len].index, df_forecast[:test_len].values,label="Predicted",color='green')
plt.plot(df[:train_len + test_len].index, df[:train_len + test_len].values,label='Actual',color='orange')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()

# ================
clfpoly2_preds = clfpoly2.predict(X_test)

myctr = 0
for i, row in df_forecast.iterrows(): 
  df_forecast.at[i,'price'] = clfpoly2_preds[myctr]
  myctr = myctr + 1    
plt.figure(figsize=(20,10))
plt.title('Quadratic Regression 2')
plt.plot(df_forecast[:test_len].index, df_forecast[:test_len].values,color='green',label="Predicted")
plt.plot(df[:train_len + test_len].index, df[:train_len + test_len].values,color='orange',label='Actual')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()

# ================
clfpoly3_preds = clfpoly3.predict(X_test)

myctr = 0
for i, row in df_forecast.iterrows(): 
  df_forecast.at[i,'price'] = clfpoly3_preds[myctr]
  myctr = myctr + 1    
plt.figure(figsize=(20,10))
plt.title('Quadratic Regression 3')
plt.plot(df_forecast[:test_len].index, df_forecast[:test_len].values,color='green',label="Predicted")
plt.plot(df[:train_len + test_len].index, df[:train_len + test_len].values,color='orange',label='Actual')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()

# ================
clfknn_preds = clfknn.predict(X_test)

myctr = 0
for i, row in df_forecast.iterrows(): 
  df_forecast.at[i,'price'] = clfknn_preds[myctr]
  myctr = myctr + 1    
plt.figure(figsize=(20,10))
plt.title('KNN Regression')
plt.plot(df_forecast[:test_len].index, df_forecast[:test_len].values,color='green',label="Predicted")
plt.plot(df[:train_len + test_len].index, df[:train_len + test_len].values,color='orange',label='Actual')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()