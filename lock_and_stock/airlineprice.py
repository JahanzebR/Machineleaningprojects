import sys
import pandas as pd
import matplotlib
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix

# Load the data

airline = pd.read_csv("airline.csv")

# Print the names of the columns in games
# print(airline.columns)
# print(airline.shape)

# Make a histogram of all the prices in the cost column
# plt.hist(airline["Cost"])
# plt.show()

# Remove any rows with missing values

airline = airline.dropna(axis=0)
# print(airline.shape)


# Create a function that isolates the number of stops from the Layover columns

def RepresentsInt(s):
    try:
        int(s)
        return int(s)
    except ValueError:
        return 0


stopnumber = []

arr = airline.Layover.to_numpy()
for i in range(len(arr)):
    stopnumber.append(RepresentsInt(arr[i].split(' ')[0]))

# print(stopnumber)

# Add the contents of stopnumber into the Data Frame
airline.insert(4, column="stopnumber", value=stopnumber)

# for i in range(10):
# print('Layover is equal to = {},'.format(airline.Layover[i]),
#     'Stop Number is equal to = {}'.format(airline.stopnumber[i]))

# Isolate the time span of the flight and simplify it to just minutes of flight time

h = airline['Time span'].str.extract('(\d+)h', expand=False).astype(float) * 60
# print(h)
m = airline['Time span'].str.extract('(\d+)m', expand=False).astype(float)

airline['flighttimeminutes'] = h.add(m, fill_value=0).astype(int)
# print(airline.flighttimeminutes[:50])
# print(airline.shape)

airline['monthofflight'] = pd.DatetimeIndex(airline['Flight Date'], dayfirst=True).month
# print(airline.monthofflight)

AirlineID = []
airlines = airline['Air Service']
airlineIds = {}
counterIds = 1

for i in airlines:
    if not airlineIds.get(i, False):
        airlineIds[i] = counterIds
        counterIds += 1

# print(airlineIds)

airservice = airline['Air Service'].to_numpy()

for i in range(len(airservice)):
    AirlineID.append(airlineIds.get(airservice[i]))

airline.insert(2, column="AirlineID", value=AirlineID)

flighthour = []
takeofftime = airline['Take off time'].to_numpy()
for i in range(len(takeofftime)):
    flighthour.append(takeofftime[i].split(':')[0])


for i in range(len(flighthour)):
    flighthour[i] = int(flighthour[i])
    flighthour[i] = flighthour[i] * 100

airline.insert(11, column="flighthour", value=flighthour)

# for i in range(10):
#     print('Airline name = {},'.format(airline['Air Service'][i]),
#           'Airline ID= {}'.format(airline.AirlineID[i]))


# plt.hist(airline["monthofflight"])
# plt.show()

# corrmat = airline.corr()
# fig = plt.figure(figsize=(12, 6))

# sns.heatmap(corrmat, vmax=0.8, square=True)
# plt.show()

# scatter plot matrix
# scatter_matrix(airline)
# plt.show()
# print(airline.columns)

columns = airline.columns.tolist()

# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Air Service", "Flight Direction/Routine", "About", "Layover", "Time span", "Cost",
                                           "Flight Date", "Start", "Stop", "Take off time", "Landing time",
                                           "flighthour", 'monthofflight']]

# Store the variable we'll be predicting on
target = "Cost"

# Generate training set
train = airline.sample(frac=0.8, random_state=1)

# Select anything not in the training set and put it in the test set
test = airline.loc[~airline.index.isin(train.index)]

# Print Shapes
# print(train.shape)
# print(test.shape)

# Import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the model class
LR = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)


# Fit the model to the training data
LR.fit(train[columns], train[target])


# Generate predictions for the test set
predictions = LR.predict(test[columns])


# Compute error between our test predictions and actual values
print('MSE using Linear Regression: {}'.format(mean_squared_error(predictions, test[target])))

# Import the random forest model
from sklearn.ensemble import RandomForestRegressor


# Initialize the model
RFR = RandomForestRegressor(n_estimators=2000, min_samples_leaf=10, random_state=1)


# Fit to the data
RFR.fit(train[columns], train[target])


# make predictions
predictions = RFR.predict(test[columns])


# compute the error between our test predictions and actual values
print('MSE using Random Forest Regression Regression: {}'.format(mean_squared_error(predictions, test[target])))

# Make prediction with both models (use reshape so the value of the test dataset passes as a 2D array )

rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1, -1))

rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1, -1))


# Print out the predictions
print(rating_LR)
print(rating_RFR)

# Actual value from the test set for comparison
print(test[target].iloc[0])
