#!/usr/bin/env python3

from dataset import load_dataset
from model import load_model


X, y = load_dataset()
estimator = load_model()
estimator.fit(X, y)
score = estimator.score(X, y)
print(f"Train score: {score}")
