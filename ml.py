from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


from abc import ABC, abstractmethod
class MLModel(ABC): 

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def gen_preprocessor(self,col):
        preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), [col])
        ],
        remainder='passthrough'
        )
        return preprocessor
    
    def calulate_metrics(self,y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse,rmse,mae,r2


class BaseLine(MLModel):

    def __init__(self):
        self.mean = None
    
    def fit(self, X, y):
        self.mean = np.mean(y)


    def predict(self, X):
        y_predict = np.full(X.shape[0], fill_value=self.mean)
        return y_predict
    
class DecisionTree(MLModel):

    def __init__(self):
        self.mean = None
        self.col = "community"
        self.pipeline = Pipeline([
            ('preprocessor', self.gen_preprocessor(self.col)),
            ('model', DecisionTreeRegressor(random_state=42))
        ])
    
    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X_test):
        y_pred = self.pipeline.predict(X_test)
        return y_pred

class RandomForest(MLModel):

    def __init__(self):
        self.col = "community"
        self.pipeline = Pipeline([
            ('preprocessor', self.gen_preprocessor(self.col)),
            ('model', RandomForestRegressor(n_estimators=100,random_state=42))
        ])
    
    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X_test):
        y_pred = self.pipeline.predict(X_test)
        return y_pred

class SVRAlg(MLModel):

    def __init__(self):
        self.mean = None
        self.col = "community"
        self.pipeline = Pipeline([
            ('preprocessor', self.gen_preprocessor(self.col)),
            ('scaler', StandardScaler(with_mean=False)),
            ('model', SVR())
        ])
    
    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X_test):
        y_pred = self.pipeline.predict(X_test)
        return y_pred

class GradientBoosting(MLModel):

    def __init__(self):
        self.mean = None
        self.col = "community"
        self.pipeline = Pipeline([
            ('preprocessor', self.gen_preprocessor(self.col)),
            ('model', GradientBoostingRegressor()),
        ])
    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X_test):
        y_pred = self.pipeline.predict(X_test)
        return y_pred

class NeuralNetwork(MLModel):
    def __init__(self):
        self.mean = None
        self.col = "community"
        model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
        self.pipeline = Pipeline([
            ('preprocessor', self.gen_preprocessor(self.col)),
            ('model', model),
        ])
    
    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X_test):
        y_pred = self.pipeline.predict(X_test)
        return y_pred

class MLFactory():
    @staticmethod
    def get_instance(algo) -> MLModel:
        if algo == "Baseline":
            return BaseLine()
        elif algo == "RandomForest":
            return RandomForest()
        elif algo == "DecisionTree":
            return DecisionTree()
        elif algo == "SVR":
            return SVRAlg()
        elif algo == "GradientBoosting":
            return GradientBoosting()
        elif algo == "NeuralNetwork":
            return NeuralNetwork()
        else:
            raise ValueError("Invalid shape type")
