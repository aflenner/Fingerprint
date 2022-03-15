import matplotlib.pyplot as plt
import numpy as np


def amplitude(omega0, Q):
    omega = np.arange(10, 11, .01)
    num = 1.0
    trm1 = omega*omega - omega0*omega0
    trm1 = trm1*trm1 
    trm2 = omega*omega*omega*omega 
    trm2 = trm2/Q/Q
    trm3 = np.sqrt(trm2 + trm1) 

    return num/trm3 

def print_peaks(Atotal, title):
    font = {'fontsize': 20}
    plt.plot(Atotal)
    plt.xlabel("Frequency", **font)
    plt.ylabel("Amplitude", **font)
    plt.title(title, **font)
    plt.xticks([],[])


def create_amplitude(frequencies):
    Q = 1000
    Atotal = 0*amplitude(10 + .5, Q)
    for f1 in frequencies:
        for f2 in frequencies:
            Atotal = Atotal + amplitude(f1, Q)*amplitude(f2,Q)

    return Atotal

numfreq = 1000
frequencies = 10 + np.random.uniform(-0.001,1.001,numfreq)
AQsmall = amplitude(10+.5, 10)
plt.figure()
print_peaks(AQsmall, "Small Q")
plt.savefig("/data/smallQ.png")
AQmedium = amplitude(10 + .5, 100)
plt.figure()
print_peaks(AQmedium, "Medium Q")
plt.savefig("/data/mediumQ.png")
AQLarge = amplitude(10 + .5, 1000)
plt.figure()
print_peaks(AQLarge, "Large Q")
plt.savefig("/data/largeQ.png")

Atotal = create_amplitude(frequencies)

plt.figure()
print_peaks(Atotal, "Resonance - Multiple Q")
frequencies = frequencies + .1*np.random.uniform(0,1,numfreq)
Atotal = create_amplitude(frequencies)
print_peaks(Atotal, "Resonance - Multiple Q")
plt.legend(['Original Frequencies', 'Perturbed Frequencies'])
plt.savefig('/data/manyStructures.png')


