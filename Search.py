# Import libraries, including GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
# Read in data from CSV
df = pd.read_csv('~/Melbourne_housing_FULL.csv')
# Delete unneeded columns
del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Lattitude']
del df['Longtitude']
del df['Regionname']
del df['Propertycount']
# Remove rows with missing values
df.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = True)
# Convert non-numeric data using one-hot encoding
df = pd.get_dummies(df, columns = ['Suburb', 'CouncilArea', 'Type'])
# Assign X and y variables
X = df.drop('Price',axis=1)
y = df['Price']
# Split data into test/train set (70/30 split) and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
shuffle = True)
# Input algorithm
model = ensemble.GradientBoostingRegressor()
122
# Set the configurations that you wish to test. To minimize processing time,
limit num. of variables or experiment on each hyperparameter separately.
hyperparameters = {
'n_estimators': [200, 300],
'max_depth': [4, 6],
'min_samples_split': [3, 4],
'min_samples_leaf': [5, 6],
'learning_rate': [0.01, 0.02],
'max_features': [0.8, 0.9],
'loss': ['ls', 'lad', 'huber']
}
# Define grid search. Run with four CPUs in parallel if applicable.
grid = GridSearchCV(model, hyperparameters, n_jobs = 4)
# Run grid search on training data
grid.fit(X_train, y_train)
# Return optimal hyperparameters
grid.best_params_
# Check model accuracy using optimal hyperparameters
mae_train = mean_absolute_error(y_train, grid.predict(X_train))
print ("Training Set Mean Absolute Error: %.2f" % mae_train)
mae_test = mean_absolute_error(y_test, grid.predict(X_test))
print ("Test Set Mean Absolute Error: %.2f" % mae_test)
