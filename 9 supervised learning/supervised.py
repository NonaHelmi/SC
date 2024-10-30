import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# بارگذاری داده‌های iris
iris = load_iris()
X = iris.data
y = iris.target

# تقسیم داده‌ها به مجموعه آموزشی و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ساخت مدل Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# آموزش مدل
model.fit(X_train, y_train)

# پیش‌بینی بر روی داده‌های تست
y_pred = model.predict(X_test)

# محاسبه دقت
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy (Supervised Learning):', accuracy)