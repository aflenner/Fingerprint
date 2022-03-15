import tensorflow as tf
from parameters import data_format
import scipy.io
import glob
import numpy as np

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_single_sample(sample, label):
  #define the dictionary -- the structure -- of our single example
  data = {
    'rows': _int64_feature(sample.shape[0]), # should be 1 or 2 - 2 for I/Q.
    'cols': _int64_feature(sample.shape[1]),
    'data': _bytes_feature(serialize_array(sample)),
    'transmitter': _int64_feature(label)
  }
  #create an Example, wrapping the single features
  out = tf.train.Example(features=tf.train.Features(feature=data))

  return out

def write_samples_to_tfr_short(samples, labels, filename:str="images"):
  filename= filename+".tfrecords"
  print("Writing to " + filename)
  writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
  count = 0

  for index in range(len(samples)):

    #get the data we want to write
    current_sample = samples[index]
    current_label = labels[index]

    out = parse_single_sample(sample=current_sample, label=current_label)
    writer.write(out.SerializeToString())
    count += 1

  writer.close()
  print(f"Wrote {count} elements to TFRecord")
  return count

def parse_tfr_element(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'rows': tf.io.FixedLenFeature([], tf.int64),
      'cols':tf.io.FixedLenFeature([], tf.int64),
      'data':tf.io.FixedLenFeature([], tf.string),
      'transmitter' : tf.io.FixedLenFeature([], tf.int64),
    }


  content = tf.io.parse_single_example(element, data)

  rows = content['rows']
  cols = content['cols']
  label = content['transmitter']
  raw_signal = content['data']


  #get our 'feature'-- our image -- and reshape it appropriately
  feature = tf.io.parse_tensor(raw_signal, out_type=tf.float64)
  feature = tf.reshape(feature, shape=[rows, cols])
  return (feature, label)

def get_dataset_small(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_tfr_element)

    return dataset

def get_airid_data(folder):

  files = glob.glob(folder + "*.mat")
  matrix_key = 'wifi_rx_data'
  try:
    data = scipy.io.loadmat(files[0])[matrix_key]
  except:
    matrix_key = 'previous_matrix'
    data = scipy.io.loadmat(files[0])[matrix_key]

  clss = 0
  if matrix_key == 'previous_matrix':
    samples = []
    labels = []
    for file in files:
      data = scipy.io.loadmat(file)[matrix_key]

      append = file.split('/')[-1]

      for row in range(data.shape[0]):
        x = np.squeeze(data[row,:])
        x = np.squeeze(x)
        samplelength = 512
        numsamples = np.int(np.floor(x.shape[0]/512))
        sample = np.zeros((2, samplelength))

        for n in range(numsamples):
          tmp = x[n*samplelength:(n+1)*samplelength]
          m = np.mean(np.abs(tmp))
          tmp = tmp/m
          sample[0,:] = np.real(tmp)
          sample[1,:] = np.imag(tmp)
          samples.append(sample)
          labels.append(clss)

      clss = clss + 1
  else:
    pass

  return samples, labels
