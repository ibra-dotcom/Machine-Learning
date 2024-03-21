#Import libraries: pandas and scikit-learn methods
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error

#Import dataset
df = pd.read_csv('~/ML/Machine Learning/Melbourne_housing_FULL.csv')

#Scrub dataset
# The misspellings of “longitude” and “latitude” are preserved here del df['Address']
del df['Method']
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

