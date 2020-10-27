import sys
import pandas
import matplotlib
import seaborn
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the data


games = pandas.read_csv("games.csv")

# Print the names of the columns in games
# print(games.columns)
# print(games.shape)


# Make a histogram of all the ratings in the average_rating column
# plt.hist(games["average_rating"])
# plt.show()


# Print the first row of all the games with zero scores
# print(games[games["average_rating"] == 0].iloc[0])

# Print the first row of games with scores greater than 0
# print(games[games["average_rating"] > 0].iloc[0])

# Remove any rows without user reviews

games = games[games["users_rated"] > 0]

# Remove any rows with missing values

games = games.dropna(axis=0)
# plt.hist(games["average_rating"])
# plt.show()

# print(games.columns)

# Correlation Matrix
corrmat = games.corr()
fig = plt.figure(figsize=(12, 9))

sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

# Get all the columns from the dataFrame
columns = games.columns.tolist()


# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

# Store the variable we'll be predicting on
target = "average_rating"

# Generate training set
train = games.sample(frac=0.8, random_state=1)

# Select anything not in the training set and put it in the test set
test = games.loc[~games.index.isin(train.index)]

# Print Shapes
# print(train.shape)
# print(test.shape)


# Import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Initialize the model class
LR = LinearRegression()


# Fit the model to the training data
LR.fit(train[columns], train[target])


# Generate predictions for the test set
predictions = LR.predict(test[columns])


# Compute error between our test predictions and actual values
mean_squared_error(predictions, test[target])


# Import the random forest model
from sklearn.ensemble import RandomForestRegressor


# Initialize the model
RFR = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)


# Fit to the data
RFR.fit(train[columns], train[target])


# make predictions
predictions = RFR.predict(test[columns])


# compute the error between our test predictions and actual values
mean_squared_error(predictions, test[target])

# Make prediction with both models (use reshape so the value of the test dataset passes as a 2D array )

rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1, -1))

rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1, -1))


# Print out the predictions
print(rating_LR)
print(rating_RFR)

# Actual value from the test set for comparison
print(test[target].iloc[0])
