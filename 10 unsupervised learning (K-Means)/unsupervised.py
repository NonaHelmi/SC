import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# تولید داده‌های تصادفی
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# اجرای K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# پیش‌بینی خوشه‌ها
y_kmeans = kmeans.predict(X)

# رسم نتایج
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering (Unsupervised Learning)')
plt.show()