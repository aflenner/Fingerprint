import tensorflow as tf

data_format = {
    'numsamples': tf.io.FixedLenFeature([], tf.int64),
    'data': tf.io.FixedLenFeature([], tf.float32),
    'transmitter': tf.io.FixedLenFeature([], tf.int64)
} 