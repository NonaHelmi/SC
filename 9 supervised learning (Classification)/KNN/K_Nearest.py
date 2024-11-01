from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


digits = load_digits()
pd.DataFrame(digits.data).head()
x = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target)

s=[]
i=range(1, 11)
for k in i:
  knn = KNeighborsClassifier(n_neighbors=k, metric='l1')
  knn.fit(X_train, y_train)
  score = knn.score(X_test,y_test)
  print("Score for K=" + str(k) + ":", score)
  s.append(score)

plt.plot(i, s)
