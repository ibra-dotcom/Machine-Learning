#Import libraries: pandas and scikit-learn methods
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error

#Import dataset
df = pd.read_csv('~/Melbourne_housing_FULL.csv')
#Scrub dataset
# The misspellings of “longitude” and “latitude” are preserved here del df['Address']
#del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Lattitude']
del df['Longtitude'] 
del df['Regionname'] 
del df['Propertycount']
df.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = True)
df = pd.get_dummies(df, columns = ['Suburb', 'CouncilArea', 'Type']
X = df.drop('Price',axis=1) y = df['Price']

#Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)
model = ensemble.GradientBoostingRegressor( n_estimators = 150,
learning_rate = 0.1,
max_depth = 30,
min_samples_split = 4, min_samples_leaf = 6, max_features = 0.6, loss = 'huber'
)
model.fit(X_train, y_train
#Evaluation
#mae_train = mean_absolute_error(y_train, model.predict(X_train)) print ("Training Set Mean Absolute Error: %.2f" % mae_train)

#Training Set Mean Absolute Error:
#mae_test = mean_absolute_error(y_test, model.predict(X_test))
print ("Test Set Mean Absolute Error: %.2f" % mae_test)

# Optimization:
#Training Set Mean Absolute Error: Should expect a value
#Test Set Mean Absolute Error:Should expect a value 
