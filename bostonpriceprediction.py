# import dependencies
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Load the Boston Housing Data set from sklearn.datasets and print it
from sklearn.datasets import load_boston
boston = load_boston()
# print(boston)

# Transform the data set into a data frame
# data=the data we want or the independent variables also known as the x values
# feature_names = the column names of the data
# target = the target variable or the price of the houses or dependent variable
# also known as the y value

df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = pd.DataFrame(boston.target)

# Get some statistics from the data set, count, mean
# print(df_x.describe())

# Initialize the linear regression model
reg = linear_model.LinearRegression()

# Import the random forest model
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
RFR = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)

# Split the data into 67% training and 33% testing data
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)

# Train the model with our training data
reg.fit(x_train, y_train)
RFR.fit(x_train, y_train.values.ravel())
params = reg.get_params()
# print(params)

# Print the coeffecients/weights for each feature/column of our model
# print(reg.coef_)

# Print the predictions on our test data
LR_y_pred = reg.predict(x_test)
# print(LR_y_pred)
RFR_y_pred = RFR.predict(x_test)
# print(RFR_y_pred)
# print the actual values
# print(y_test)

# Check the model performance/accuracy using Mean Squared Error (MSE)
# print(np.mean((y_pred - y_test)**2))

# Check the model performance/accuracy using sklearn.metrics
from sklearn.metrics import mean_squared_error
LR_prediction_error = mean_squared_error(y_test, LR_y_pred)
RFR_prediction_error = mean_squared_error(y_test, RFR_y_pred)
print('Mean Squared Error obtained by performing Linear Regression: {}'.format(LR_prediction_error))
print('Mean Squared Error obtained by performing Random Forest Regression Regression: {}'.format(RFR_prediction_error))
