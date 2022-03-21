import dataHandling as dh
import glob
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import models
import training 

base_folder = "/data/AirID_Globecom2020_dataset/"
extra = "20 Sept_ParkingLotISEC_data/"
checkpoint_path = "/checkpoints"
# # extra = "20 Sept_ParkingLotISEC_data/"
folder = base_folder + extra
#
samples, labels = dh.get_airid_data(folder)

dh.write_samples(samples, labels, '/data/parkinglot')
dataset = dh.get_dataset('/data/test.tfrecords')
dataset = dataset.shuffle(100000, reshuffle_each_iteration=True).batch(128, drop_remainder=True)
model = models.vgg()

training.train(model, dataset)