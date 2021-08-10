import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# data load
housing = pd.read_csv("sample_data/california_housing_train.csv")
housing_ind = housing.drop("median_house_value",axis=1)
housing_dep = housing["median_house_value"]

# splitting data
X_train,X_test,y_train,y_test = train_test_split(housing_ind, housing_dep, test_size=0.2,random_state=42)

# scaling data
independent_scaler = StandardScaler()
X_train = independent_scaler.fit_transform(X_train)
X_test = independent_scaler.transform(X_test)

# fitting model
linearRegModel = LinearRegression()
linearRegModel.fit(X_train,y_train)

# testing code

ACCEPTABLE_ERROR_THRESHOLD = 5000

def test_acceptable_error(model, X_test, y_test):
  y_pred = linearRegModel.predict(X_test)
  print( np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
  assert ACCEPTABLE_ERROR_THRESHOLD >= np.sqrt(metrics.mean_squared_error(y_test,y_pred))

test_acceptable_error(linearRegModel, X_test, y_test)