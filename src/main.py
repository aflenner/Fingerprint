import create_tfrecord as cr 
import glob
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

base_folder = "/data/AirID_Globecom2020_dataset/"
extra = "KRI_ControlledHovering_data/"
# extra = "20 Sept_ParkingLotISEC_data/"
folder = base_folder + extra
# matrix_key = 'wifi_rx_data'
matrix_key = 'previous_matrix'

files = glob.glob(folder + "*.mat")

data = scipy.io.loadmat(files[0])[matrix_key]
print(data.shape)

samples, labels = cr.get_airid_data(folder)
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