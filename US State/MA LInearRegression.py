import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

df=pd.read_csv('MA.csv')
values = {'hospitalizedCurrently': 0, 'positive': 0,'pending':0 ,'negative':0}
df.fillna(value=values,inplace=True)

mse_list = []

# List of all predictor combinations to fit the curve
cols = [['positive'], ['pending'], ['hospitalizedCurrently'], ['positive', 'pending'], ['positive', 'hospitalizedCurrently'], ['pending', 'hospitalizedCurrently'],
        ['positive', 'pending', 'hospitalizedCurrently']]

for i in cols:
    # Set each of the predictors from the previous list as x
    x = df[i]

    # Normalise the data
    x = preprocessing.normalize(x)

    # "Sales" column is the reponse variable
    y = df['negative']

    # Normalizing the response variable
    y = np.interp(y, (y.min(), y.max()), (0, +1))

    # Splitting the data into train-test sets with 80% training data and 20% testing data.
    # Set random_state as 0
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=0)

    # Create a LinearRegression object and fit the model
    lreg = LinearRegression()
    lreg.fit(xtrain, ytrain)

    # Predict the response variable for the test set
    y_pred = lreg.predict(xtest)

    # Compute the MSE
    MSE = mean_squared_error(y_pred, ytest)

    # Append the MSE to the list
    mse_list.append(MSE)

t = PrettyTable(['Predictors', 'MSE'])

#Loop to display the predictor combinations along with the MSE value of the corresponding model
for i in range(len(mse_list)):
    t.add_row([cols[i],mse_list[i]])

print(t)
