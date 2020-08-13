import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('MA.csv')
# Get the column values for x & y
values = {'hospitalizedCurrently': 0, 'recovered': 0}
df.fillna(value=values,inplace=True)

x = df[['hospitalizedCurrently']].iloc[:50].values
y = df[['recovered']].iloc[:50].values
print(x)
print(y)

fig, ax = plt.subplots()
ax.plot(x,y,'x')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.set_title('$y$ vs $x$');
plt.show()

xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.8,random_state=60)
print(xtrain)
### edTest(test_regression) ###
# To iterate over the range, select the maximum degree of the polynomial
maxdeg = 29

# Create two empty lists to store training and testing MSEs
training_error, testing_error = [], []

# Run a for loop through the degrees of the polynomial, fit linear regression, predict y values and calculate the training and testing errors and update it to the list
for d in range(maxdeg):
    # Use polynomial features function on training & testing values
    Xtrain = PolynomialFeatures(d).fit_transform(xtrain)
    Xtest = PolynomialFeatures(d).fit_transform(xtest)

    est = LinearRegression()
    est.fit(Xtrain, ytrain)

    # Use the model predict function on training and testing values
    ytrain_pred = est.predict(Xtrain)
    ytest_pred = est.predict(Xtest)

    # Save the mean_squared
    training_error.append(mean_squared_error(ytrain_pred, ytrain))

    testing_error.append(mean_squared_error(ytest_pred, ytest))

min_error_val = min(testing_error)

# Find the degree associated with this minimum value
best_d = testing_error.index(min_error_val)
print(best_d)
fig, ax = plt.subplots()
# Using the best degree from above Make polyomial features using entire dataset (all values of x)
X = PolynomialFeatures(best_d).fit_transform(x)

# Create a linear regression object & fit it on training data
est = LinearRegression()
est.fit(X,y)

# Using the model from above, predict values for y
y_pred = est.predict(X)


# Using the model from above, predict values for y
fig, ax = plt.subplots(figsize = (10,4))
ax.plot(x,y, '.', label = 'Observed values')
ax.plot(x, y_pred, 's', color = 'r', alpha = 0.3, ms = 4, label = 'Best fit model')

ax.set_xlabel('$X$',fontsize=16)
ax.set_ylabel('$Y$',fontsize=16)
ax.legend(loc = 'best');
plt.show()