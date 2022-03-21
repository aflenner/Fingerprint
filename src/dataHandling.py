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

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def divide_by_mean(sample, label):
  tmp = tf.math.multiply(sample, sample)
  tmp = sample[0,:] + sample[1,:]
  tmp = tf.math.sqrt(tmp)
  m = tf.reduce_mean(tmp)
  sample = sample/m 
  return sample, label

def parse_single_example(data, label):
    # make train example
    example = tf.train.Example(features=tf.train.Features(
        feature={
        'rows': _int64_feature(data.shape[0]),
        'cols': _int64_feature(data.shape[1]),
        'label': _int64_feature(label),
        'data': _bytes_feature(serialize_array(data))
        }))

    return example

def write_samples(samples, labels, filename:str="images"):
  filename= filename+".tfrecords"
  print("Writing to " + filename)
  writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
  count = 0

  for index in range(len(samples)):

    #get the data we want to write
    current_sample = samples[index]
    current_label = labels[index]

    out = parse_single_example(current_sample, current_label)
    writer.write(out.SerializeToString())
    count += 1

  writer.close()
  print(f"Wrote {count} elements to TFRecord")
  return count

def parse_element(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'rows': tf.io.FixedLenFeature([], tf.int64),
      'cols':tf.io.FixedLenFeature([], tf.int64),
      'label':tf.io.FixedLenFeature([], tf.int64),
      'data' : tf.io.FixedLenFeature([], tf.string),
    }


  content = tf.io.parse_single_example(element, data)

  rows = content['rows']
  cols = content['cols']
  label = content['label']
  raw_signal = content['data']

  feature = tf.io.parse_tensor(raw_signal, out_type=tf.float64)
  feature = tf.reshape(feature, shape=[rows, cols])
  
  return (feature, label)

def get_dataset(filename):
    print("Loading " + filename)
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_element)

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

      for row in range(data.shape[0]):
        x = np.squeeze(data[row,:])
        x = np.squeeze(x)
        samplelength = 256
        numsamples = np.int(np.floor(x.shape[0]/(samplelength)))
        sample = np.zeros((samplelength, 2))

        for n in range(numsamples):
          tmp = x[n*samplelength:(n+1)*samplelength]
        #   m1 = np.max(np.abs(np.real(tmp)))
        #   m2 = np.max(np.abs(np.imag(tmp)))
        #   m = np.max(np.array([m1,m2]))
        #   tmp = tmp + (m + 1j*m) 
        #   tmp = tmp/(2*m)
          tmp = tmp - np.mean(tmp)
          tmp = tmp/np.std(tmp)
          sample[:, 0] = np.real(tmp)
          sample[:, 1] = np.imag(tmp)
        #   sample[:, 0] = sample[:,0] - np.mean(sample[:,0])
        #   sample[:, 0] = sample[:,0]/np.std(sample[:,0])
        #   sample[:, 1] = sample[:,1] - np.mean(sample[:,1])
        #   sample[:, 1] = sample[:,1]/np.std(sample[:,1])
            
          samples.append(sample)
          labels.append(clss )

      clss = clss + 1
  else:
    pass

  # samples = np.array(samples)
  # labels = np.array(labels)
  return samples, labels
