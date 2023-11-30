# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as M
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import fire

# Загрузка данных
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()

# one-hot encode для ответов
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_val_oh = keras.utils.to_categorical(y_val, 10)

# Сборка модели
K.clear_session()
model = M.Sequential()
model.add(L.Conv2D(16, kernel_size=3, strides=1, padding="same", input_shape=(28, 28, 1)))
model.add(L.MaxPool2D())
model.add(L.Conv2D(32, kernel_size=3, strides=1, padding="same"))
model.add(L.MaxPool2D())
model.add(L.Conv2D(64, kernel_size=3, strides=1, padding="same"))
model.add(L.MaxPool2D())
model.add(L.Flatten())
model.add(L.Dense(10, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Центрирование и нормирование
x_train_float = x_train.astype(np.float) / 255 - 0.5
x_val_float = x_val.astype(np.float) / 255 - 0.5

# Обучение модели
model.fit(
    x_train_float[:, :, :, np.newaxis],
    y_train_oh,
    batch_size=32,
    epochs=5,
    validation_data=(x_val_float[:, :, :, np.newaxis], y_val_oh),
)
if __name__ == "__main__":
    fire.Fire()
