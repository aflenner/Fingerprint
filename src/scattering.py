import numpy as np 
from kymatio.sklearn import Scattering1D
import matplotlib.pyplot as plt
import scipy.io 
import glob


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


T = 2 ** 13
x = generate_harmonic_signal(T)
print(x.shape)
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

folder = "/data/AirID_Globecom2020_dataset/20 Sept_ParkingLotISEC_data/"

files = glob.glob(folder + "*.mat")

cnt = 0
data = scipy.io.loadmat(files[0])['previous_matrix']
x = data[0,0:2**15]
Sx, meta = calc_scattering(x)
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)
Sx0 = Sx[order0][0]
Sx1 = Sx[order1]
Sx2 = Sx[order2]
for file in files:
    data = scipy.io.loadmat(file)['previous_matrix']

    x = data[0,0:2**15]
    x = np.squeeze(x)
    Sx, meta = calc_scattering(x)
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)

    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.plot(Sx[order0][0] )
    plt.title('Zeroth-order scattering')
    plt.subplot(3, 1, 2)
    plt.imshow(Sx[order1] , aspect='auto')
    plt.title('First-order scattering')
    plt.subplot(3, 1, 3)
    plt.imshow(Sx[order2] , aspect='auto')
    plt.title('Second-order scattering')
    plt.savefig("/results/scattering" + str(cnt) + ".png")
    cnt = cnt + 1