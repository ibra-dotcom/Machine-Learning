# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
# Read in data from CSV
df = pd.read_csv('~/Downloads/Melbourne_housing_FULL.csv')
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
# Set up algorithm
model = ensemble.GradientBoostingRegressor(
n_estimators = 250,
learning_rate = 0.1,
max_depth = 5,
120
min_samples_split = 4,
min_samples_leaf = 6,
max_features = 0.6,
loss = 'huber'
)
# Run model on training data
model.fit(X_train, y_train)


# Vérification accuracy (up to two decimal places)
mae_train = mean_absolute_error(y_train, model.predict(X_train))
print ("Training Set Mean Absolute Error: %.2f" % mae_train)
mae_test = mean_absolute_error(y_test, model.predict(X_test))
print ("Test Set Mean Absolute Error: %.2f" % mae_test)



