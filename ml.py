from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def baseline(y_train,y_test):
    mean_target = np.mean(y_train)
    y_baseline = np.full_like(y_test, fill_value=mean_target)
    mse_baseline = mean_squared_error(y_test, y_baseline)
    return mse_baseline

def decisiontree(X_train, y_train,X_test,y_test):
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse



def randomforest(X_train, y_train,X_test,y_test):
    rf_classifier = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    mse_random = mean_squared_error(y_test, y_pred)
    return mse_random
