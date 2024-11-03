import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

# تعریف داده‌ها (رنگ‌ها به صورت RGB)
colors = np.array([
    [0., 0., 0.],       # Black
    [0., 0., 1.],       # Blue
    [0., 1., 0.],       # Green
    [1., 0., 0.],       # Red
    [1., 1., 0.],       # Yellow
    [1., 0., 1.],       # Magenta
    [0., 1., 1.],       # Cyan
    [1., 1., 1.]        # White
])

# ایجاد SOM
som_size = (3, 3)  # اندازه شبکه SOM
som = MiniSom(som_size[0], som_size[1], input_len=3, sigma=1.0, learning_rate=0.5)

# آموزش SOM
som.train_random(colors, num_iteration=100)

# بصری‌سازی نتایج
plt.figure(figsize=(7, 7))
for i in range(som_size[0]):
    for j in range(som_size[1]):
        # دریافت وزن‌های هر نود
        weight = som.get_weights()[i, j]
        plt.text(j, i, f'{weight}', ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, lw=0))
        plt.scatter(j, i, color=weight)

plt.title('Self-Organizing Map of Colors')
plt.xticks(range(som_size[1]))
plt.yticks(range(som_size[0]))
plt.grid()
plt.show()