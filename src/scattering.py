import numpy as np 
from kymatio.sklearn import Scattering1D
import matplotlib.pyplot as plt
import scipy.io 
import glob
# import Tensorflow as tf

def generate_harmonic_signal(T, num_intervals=4, gamma=0.9, random_state=42):
    """
    Generates a harmonic signal, which is made of piecewise constant notes
    (of random fundamental frequency), with half overlap
    """
    rng = np.random.RandomState(random_state)
    num_notes = 2 * (num_intervals - 1) + 1
    support = T // num_intervals
    half_support = support // 2

    base_freq = 0.1 * rng.rand(num_notes) + 0.05
    phase = 2 * np.pi * rng.rand(num_notes)
    window = np.hanning(support)
    x = np.zeros(T, dtype='float32')
    t = np.arange(0, support)
    u = 2 * np.pi * t
    for i in range(num_notes):
        ind_start = i * half_support
        note = np.zeros(support)
        for k in range(1):
            note += (np.power(gamma, k) *
                     np.cos(u * (k + 1) * base_freq[i] + phase[i]))
        x[ind_start:ind_start + support] += note * window

    return x

def calc_scattering(x):
    J = 6
    Q = 16
    T = x.shape[0]

    scattering = Scattering1D(J, T, Q)
    meta = scattering.meta()

    Sx = scattering(x)

    return Sx, meta


# T = 2 ** 13
# x = generate_harmonic_signal(T)
# print(x.shape)
# plt.figure(figsize=(8, 2))
# plt.plot(x)
# plt.title("Original signal")
# plt.savefig("/results/Example.png")

# J = 6
# Q = 16

# scattering = Scattering1D(J, T, Q)

# meta = scattering.meta()
# order0 = np.where(meta['order'] == 0)
# order1 = np.where(meta['order'] == 1)
# order2 = np.where(meta['order'] == 2)

# Sx = scattering(x)

# plt.figure(figsize=(8, 8))
# plt.subplot(3, 1, 1)
# plt.plot(Sx[order0][0])
# plt.title('Zeroth-order scattering')
# plt.subplot(3, 1, 2)
# plt.imshow(Sx[order1], aspect='auto')
# plt.title('First-order scattering')
# plt.subplot(3, 1, 3)
# plt.imshow(Sx[order2], aspect='auto')
# plt.title('Second-order scattering')
# plt.savefig("/results/scattering.png")

base_folder = "/data/AirID_Globecom2020_dataset/"
extra = "17Sept_KRI_data/7feetbadhovering/"
folder = base_folder + extra
matrix_key = 'wifi_rx_data'
matrix_key = 'previous_matrix'

files = glob.glob(folder + "*.mat")
print(files[0])

cnt = 1
data = scipy.io.loadmat(files[0])[matrix_key]
print(data)
plt.figure()
# for file in files:
#     data = scipy.io.loadmat(file)[matrix_key]
#     x = np.squeeze(data)
#     x = np.abs(np.fft.fftshift(np.fft.fft(x)))
#     # x = x/np.linalg.norm(x, axis=0)
#     append = file.split('/')[-1]
#     plt.subplot(len(files),1, cnt)
#     # plt.plot(np.real(x), np.imag(x), 'b*')
#     plt.plot(x)
#     # plt.xlim([-1.1,1.1])
#     # plt.ylim([-1.1,1.1])
#     plt.xticks([])
#     plt.yticks([])
#     cnt = cnt + 1
# plt.savefig('/results/Last_over_air_transmissionB200Txffft.png')
figrows = len(files)
figcols = data.shape[0]
for file in files:
    data = scipy.io.loadmat(file)[matrix_key]
    append = file.split('/')[-1]

    for row in range(data.shape[0]):
        x = np.squeeze(data[row,:])
        x = x[1000:1500]
        # x = np.abs(np.fft.fftshift(np.fft.fft(x)))
        plt.subplot(figrows, figcols, cnt)
        plt.plot(np.real(x), np.imag(x), 'b*')
        # plt.plot(x)
        plt.xticks([])
        plt.yticks([])
        
        cnt = cnt + 1

plt.savefig('/results/7feetbadhovering.png')
# for file in files:
#     data = scipy.io.loadmat(file)[matrix_key]

#     x = data[1,0:2**10]
#     x = np.squeeze(x)
#     Sx, meta = calc_scattering(x)
#     order0 = np.where(meta['order'] == 0)
#     order1 = np.where(meta['order'] == 1)
#     order2 = np.where(meta['order'] == 2)

#     plt.figure(figsize=(8, 8))
#     plt.subplot(3, 1, 1)
#     plt.plot(np.real(x), np.imag(x), 'b*')
#     plt.title('Zeroth-order scattering')
#     plt.subplot(3, 1, 2)
#     # plt.imshow(Sx[order1] , aspect='auto')
#     plt.imshow(Sx[order1], aspect = 'auto')
#     plt.title('First-order scattering')
#     plt.subplot(3, 1, 3)
#     plt.imshow(Sx[order2] , aspect='auto')
#     plt.title('Second-order scattering')
#     plt.savefig("/results/scattering1" + str(cnt) + ".png")
#     cnt = cnt + 1

# for file in files:
#     data = scipy.io.loadmat(file)[matrix_key]

#     x = data[2,0:2**10]
#     x = np.squeeze(x)
    
#     Sx, meta = calc_scattering(x)
#     order0 = np.where(meta['order'] == 0)
#     order1 = np.where(meta['order'] == 1)
#     order2 = np.where(meta['order'] == 2)

#     plt.figure(figsize=(8, 8))
#     plt.subplot(3, 1, 1)
#     plt.plot(np.real(x), np.imag(x), 'b*')
#     plt.title('Zeroth-order scattering')
#     plt.subplot(3, 1, 2)
#     # plt.imshow(Sx[order1] , aspect='auto')
#     plt.imshow(Sx[order1], aspect='auto')
#     plt.title('First-order scattering')
#     plt.subplot(3, 1, 3)
#     plt.imshow(Sx[order2] , aspect='auto')
#     plt.title('Second-order scattering')
#     plt.savefig("/results/scattering2" + str(cnt) + ".png")
#     cnt = cnt + 1