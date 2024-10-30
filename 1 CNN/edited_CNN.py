import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import kagglehub

# مسیر داده‌ها
data_dir = kagglehub.dataset_download("balraj98/apple2orange-dataset")

# تنظیمات
img_height, img_width = 150, 150
batch_size = 32

# ایجاد دیتاجنراتور برای بارگذاری تصاویر
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2  # 20% داده‌ها برای اعتبارسنجی
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # دو کلاس: سیب و پرتقال
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # دو کلاس: سیب و پرتقال
    subset='validation'
)

# ساخت مدل CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # برای دو کلاس از سیگموید استفاده می‌کنیم
])

# کامپایل مدل
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# آموزش مدل
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10  # تعداد اپوک‌ها بر اساس نیاز
)

# ارزیابی مدل
plt.plot(history.history['accuracy'], label='دقت آموزش')
plt.plot(history.history['val_accuracy'], label='دقت اعتبارسنجی')
plt.title('دقت مدل')
plt.xlabel('اپوک')
plt.ylabel('دقت')
plt.legend()
plt.show()
