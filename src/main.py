import create_tfrecord as cr
import glob
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

base_folder = "/data/AirID:Globecom2020_dataset/"
extra = "20 Sept_ParkingLotISEC_data/"
# # extra = "20 Sept_ParkingLotISEC_data/"
# folder = base_folder + extra
#
# samples, labels = cr.get_airid_data(folder)
#
# cr.write_samples_to_tfr_short(samples, labels, "/data/parkingLot")
# dataset_small = cr.get_dataset_small("/data/parkingLot.tfrecords")
#
# data_ = list(dataset_small.as_numpy_iterator())
# print(len(data_))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# data you would like to save, dtype=float32
data = np.empty(shape=(5, 1))
label = 1
for k in range(5):
    data[k] = 1.0

# open tfrecord file
writer = tf.io.TFRecordWriter('results/test.tfrecord')

# make train example
example = tf.train.Example(features=tf.train.Features(
    feature={
    'label': _int64_feature(label),
    'data': _floats_feature(data)
    }))

# write on the file
writer.write(example.SerializeToString())

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
