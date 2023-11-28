# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

# Загрузка данных
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()

# Визуализация данных
x_train_image = np.tile(x_train[5, :, :].reshape((28, 28))[:, :, np.newaxis], (1, 1, 3))
plt.imshow(x_train_image[15:20, 5:10], cmap="Greys")
plt.show()
plt.imshow(x_train_image, cmap="Greys")
plt.show()
print("y_train [shape %s] 10 примеров:\n" % (str(y_train.shape)), y_train[:10])

# one-hot encode для ответов
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_val_oh = keras.utils.to_categorical(y_val, 10)
print(y_train_oh.shape)
print(y_train_oh[:5], y_train[:5])