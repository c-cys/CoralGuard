import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
from PIL import Image

def outout(inputs, num_units, dropout_rate=0.3, training=True):

    if training:
        inputs = tf.nn.dropout(inputs, rate=dropout_rate)

    return tf.reduce_max(tf.reshape(inputs, [-1, num_units, 2]), axis=-1)

class YooSeongActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(YooSeongActivation, self).__init__(**kwargs)
    def call(self, inputs):
        return (tf.tanh(inputs) ** 2) * tf.exp(inputs) / (tf.exp(inputs) + 1)

def CleverLoss(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    intersection = tf.reduce_sum(tf.minimum(y_true, y_pred), axis=-1)
    union = tf.reduce_sum(tf.maximum(y_true, y_pred), axis=-1)
    iou = intersection / (union + 1e-7)
    loss = 1 - iou
    return tf.reduce_mean(loss)

class SelfAttention(layers.Layer):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.softmax = layers.Softmax(axis=-1)

    def call(self, inputs):
        # Query, Key, Value 계산
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Attention Scores 계산
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_weights = self.softmax(attention_scores)

        # Attention Output 계산
        attention_output = tf.matmul(attention_weights, value)
        return attention_output

model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    SelfAttention(embed_dim=32),

    layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    SelfAttention(embed_dim=64),

    layers.Flatten(),

    layers.Dense(6 * 24 * 24 * 2),
    layers.Lambda(lambda x: outout(x, 6 * 24 * 24, dropout_rate=0.2)),

    layers.Dense(2),
    YooSeongActivation()
])

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.load_weights('weights/best_model.keras')

test_images = []
folder = "assets/coral_located"
for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize((64, 64))
            test_images.append(img)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
print(test_images)

y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)