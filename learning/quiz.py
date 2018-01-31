import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

import utils

data = pd.read_csv('./data.csv')

X = np.array(data[['x1','x2']])
y = np.array(data ['y'])

np.random.seed(55)

X2, y2 = utils.randomize(X, y)

### Logistic Regression
estimator = LogisticRegression()
utils.draw_learning_curves(X2, y2, estimator, 10)

### Decision Tree
estimator = GradientBoostingClassifier()
utils.draw_learning_curves(X2, y2, estimator, 10)

### Support Vector Machine
estimator = SVC(kernel='rbf', gamma=1000)
utils.draw_learning_curves(X2, y2, estimator, 10)

