from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from .deepMlp_network import DeepMLP
import numpy as np

# بارگذاری داده‌های Iris
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# تبدیل برچسب‌ها به فرمت One-Hot
encoder = OneHotEncoder(sparse_output=False)  # تغییر sparse به sparse_output
y_onehot = encoder.fit_transform(y)

# تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# ایجاد و آموزش شبکه عصبی عمیق
dnn = DeepMLP(layers=[4, 5, 3], learning_rate=0.1, n_iter=10000)
dnn.fit(X_train, y_train)

# پیش‌بینی بر روی داده‌های آزمایشی
predictions = dnn.predict(X_test)

# تبدیل پیش‌بینی‌ها به برچسب‌های کلاس
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# محاسبه دقت
accuracy = np.mean(predicted_classes == true_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")