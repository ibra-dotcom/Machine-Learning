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
