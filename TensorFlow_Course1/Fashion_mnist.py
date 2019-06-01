# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
load_data = tf.keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = load_data.load_data()
import matplotlib.pyplot as plt
plt.imshow(train_x[0])
print(train_x[0])
print(train_y[0])
train_x = train_x / 255.0
test_x = test_x / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=5)
model.evaluate(test_x, test_y)
#%%
print(np.argmax(model.predict(test_x[1].reshape(1,28,28,1))))
plt.imshow(test_x[1])
print(test_x[1].reshape(1,28,28,1).shape)
print(test_x[2].shape)