import numpy as np
import matplotlib.pyplot as plt
from MLL_functions import getpulse, getspectrum

plt.rc('font', size=15)

# Load the data
dataA = np.loadtxt('output/output_testOutput.txt', dtype=complex)

timeStep = 1.035e-13
lambda0 = 1.552e-6

timeA = np.arange(len(dataA)) * timeStep

plt.plot(timeA, np.abs(dataA)**2)
plt.xlabel('Time [s]')
plt.ylabel('Power [W]')
plt.show()

[valid, lambdas, spectrum, lambdas_envelope, envelope, lambdas_phases, phases] = getspectrum(timeStep, lambda0, dataA, get_envelope_and_phase=True)
if valid == True:
    freq_center = int(np.where(spectrum == max(spectrum))[0])
    lambda_center = lambdas[freq_center]
    plt.plot(lambdas_envelope, envelope)
    # plt.xlim([lambda_center - 5, lambda_center + 5])
    # plt.ylim([np.max(spectrum) - 20, 5 + np.max(spectrum)])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Amplitude [10 dB per div]')
    yticks = np.arange(np.min(envelope), np.max(envelope) + 5, step=10)
    plt.yticks(yticks, labels=[' ' for i in range(len(yticks))])
plt.show()