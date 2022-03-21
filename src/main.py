import dataHandling as dh
import glob
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import models

base_folder = "/data/AirID_Globecom2020_dataset/"
extra = "20 Sept_ParkingLotISEC_data/"
# # extra = "20 Sept_ParkingLotISEC_data/"
folder = base_folder + extra
#
samples, labels = dh.get_airid_data(folder)
print(samples[0])

dh.write_samples(samples, labels, '/data/test')
dataset = dh.get_dataset('/data/test.tfrecords')
EPOCHS = 10

# dataset = dataset.map(dh.divide_by_mean)
dataset = dataset.shuffle(100000, reshuffle_each_iteration=True).batch(128, drop_remainder=True)
# batched_dataset = dataset.batch(7, drop_remainder=True)
print(dataset)

model = models.vgg()
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
# for sample in dataset.take(1):
#     print(model(sample[0]))

# signal, label = next(iter(dataset))
# print(signal)
# write_samples(samples, labels, '/results/test')
# print(data)
# example = parse_single_example(samples[0], labels[0])
# write on the file
# writer.write(example.SerializeToString())

# cnt = 1
# figrows = len(files)
# figcols = data.shape[0]
# for file in files:
#     data = scipy.io.loadmat(file)[matrix_key]

#     append = file.split('/')[-1]

#     for row in range(data.shape[0]):
#         x = np.squeeze(data[row,:])
#         x = x[0:500]
#         m = np.mean(np.abs(x))
#         print(m)
#         # x = np.abs(np.fft.fftshift(np.fft.fft(x)))
#         plt.subplot(figrows, figcols, cnt)
#         plt.plot(np.real(x)/m, np.imag(x)/m, 'b*')
#         # plt.plot(x)
#         # plt.xticks([])
#         # plt.yticks([])

#         cnt = cnt + 1

# plt.savefig('/results/controlledhovering.png')
