import create_tfrecord as cr
import glob
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

base_folder = "/data/AirID:Globecom2020_dataset/"
extra = "20 Sept_ParkingLotISEC_data/"
# extra = "20 Sept_ParkingLotISEC_data/"
folder = base_folder + extra

samples, labels = cr.get_airid_data(folder)

# cr.write_samples_to_tfr_short(samples, labels, "/data/parkingLot")
dataset_small = cr.get_dataset_small("/data/parkingLot.tfrecords")

print(dataset_small)
data = list(dataset_small.as_numpy_iterator())



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
