from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_model():
estimator = Pipeline([("std", StandardScaler()),
("lr", SGDRegressor())])
return estimator
