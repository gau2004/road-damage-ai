"""
model_builder.py
Builds a MobileNetV2-based transfer-learning model for
binary road-damage classification (Broken Road / Not Broken Road).
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models


def build_model(image_size=224, activation='relu', dropout=0.3,
                 learning_rate=0.001, optimizer_name='adam'):
    """
    Build and compile a MobileNetV2 transfer-learning model.

    Args:
        image_size (int): input image width/height (square images).
        activation (str): activation function for the dense layer
                           ('relu', 'tanh', 'sigmoid').
        dropout (float): dropout rate before the output layer.
        learning_rate (float): optimizer learning rate.
        optimizer_name (str): 'adam', 'sgd', or 'rmsprop'.

    Returns:
        tf.keras.Model: compiled model ready for training.
    """
    base_model = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # freeze the pretrained backbone

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=activation)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name='road_damage_mobilenetv2')

    optimizer_name = (optimizer_name or 'adam').lower()
    if optimizer_name == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
