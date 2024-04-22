from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from abc import ABC, abstractmethod
class MLModel(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def gen_preprocessor(col):
        preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), [col])
        ],
        remainder='passthrough'
        )
        return preprocessor
    
    def gen_pipeline(col):
        pipeline = Pipeline([
            ('preprocessor', gen_preprocessor(col)),
            ('model', RandomForestRegressor(random_state=42))
        ])


class BaseLineRegressor(MLModel):

    def __init__(self):

        self.mean = None
    
    def fit(self, X, y):
        self.mean = np.mean(y)


    def predict(self, X):
        y_predict = np.full_like(X, fill_value=self.mean)
        return y_predict

