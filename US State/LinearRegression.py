import csv
import numpy
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
data=read_csv('../../Us State/AI_HUB/us_daily.csv')
plt.scatter(data['positive'],data['posNeg'])
plt.show()
lrModel=LinearRegression()
x=data[['positive']]
y=data[['posNeg']]
lrModel.fit(x,y);
print(lrModel.score(x,y))
