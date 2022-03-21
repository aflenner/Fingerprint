import dataHandling as dh
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import models

def train(model, dataset):

    EPOCHS = 10

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=.0001)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(signals, labels):
        with tf.GradientTape() as tape:
            predictions = model(signals, training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_accuracy(labels, predictions)

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for signals, labels in dataset:
            train_step(signals, labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result()*100 } '
        )

    model.summary()
    model.save('/models/vgg_model')