#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import cross_validate

from dataset import load_dataset
from model import load_model

X, y = load_dataset()
estimator = load_model()
cv_scores = cross_validate(estimator, X, y)
estimator.fit(X, y)
score = estimator.score(X, y)
print(f"Train score: {score}")
print(f"CV score: {np.mean(cv_scores['test_score'])} ({cv_scores['test_score']})")