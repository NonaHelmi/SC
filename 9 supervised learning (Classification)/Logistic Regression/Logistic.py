import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
x = iris.data[:, 3:] # petal width
print(iris.target)
y = np.where(iris.target == 2, 1, 0)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
test_size=0.3, random_state=42, stratify=iris.target)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
print(log_reg.score(x_test, y_test) )

y_proba = log_reg.predict_proba(x_test[:3])
print(y_proba)

y_predict = log_reg.predict(x_test[:3])
print(y_predict)
