import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as mse


df=pd.read_csv('us_daily2.csv')
x=df[['date', 'death','totalTestResultsIncrease', 'negativeIncrease','totalTestResults','negative']]
y=df['positive']
yy=df.positive.values
pre=[]
x = preprocessing.normalize(x)
y = np.interp(y, (y.min(), y.max()), (0, +1))
print(x)
for i in range(0,len(x)-14):
    x1=x[i:i+13]
    y1=y[i:i+13]
    lr=LinearRegression()
    lr.fit(x1,y1)
    y_pred=lr.predict(x[i+14].reshape(1,-1))
    pre.append(y_pred)
print(pre)
print(len(pre))
print(len(x[14:]))
print(y[14:])
plt.figure(1)
plt.scatter(range(14,len(pre)+14),pre,marker='x')
plt.savefig('prediction.jpg')
plt.figure(2)
plt.scatter(range(14,len(pre)+14),y[14:],marker='x')
plt.savefig('reality.jpg')
plt.show()
print(mse(pre,y[14:]))