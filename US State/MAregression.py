# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
# cmap(颜色)

import seaborn as sns
import numpy as np
import pandas as pd
df=pd.read_csv('MA.csv')
values = {'hospitalizedCumulative': 0, 'inIcuCumulative': 0,'onVentilatorCumulative':0 }
df.fillna(value=values,inplace=True)
X = df[['hospitalizedCumulative','inIcuCumulative','onVentilatorCumulative']]
x = preprocessing.normalize(X)
y = df['positive']

X_train, X_val, y_train, y_val = train_test_split(x,y,test_size=0.3,random_state=31)
lreg = LinearRegression()
lreg.fit(X_train, y_train)

# Predict on the validation set

yval_pred = lreg.predict(X_val)
mse = mean_squared_error(yval_pred,y_val)

# print the MSE value

print ("Multi-linear regression MSE is", mse)