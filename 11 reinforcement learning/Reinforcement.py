import gym
import numpy as np
import random
from IPython.display import clear_output

# ایجاد محیط Taxi-v3
env = gym.make("Taxi-v3")

# مقداردهی اولیه Q-Table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# پارامترهای یادگیری
alpha = 0.1  # نرخ یادگیری
gamma = 0.6  # فاکتور تنزیل
epsilon = 0.1  # احتمال انتخاب عمل تصادفی

# لیست برای ذخیره نتایج
all_epochs = []
all_penalties = []

# آموزش مدل
for i in range(1, 100001):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            # انتخاب عمل تصادفی
            action = env.action_space.sample()
        else:
            # انتخاب بهترین عمل بر اساس Q-Table
            action = np.argmax(q_table[state])

        # اجرای عمل و دریافت نتایج
        next_state, reward, done, info = env.step(action)

        # به‌روزرسانی Q-Table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # فرمول به‌روزرسانی Q-Table
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:  # مجازات برای عمل نامناسب
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.")