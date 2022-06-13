import numpy as np
import matplotlib.pyplot as plt
from MLL_functions import getpulse, getspectrum

plt.rc('font', size=15)

# Load the data

def plotVerticalLine(x, height):
    plt.plot(x * np.ones_like((1,2)), np.array([0.0, height]))

timeStep = 1.035e-13
lambda0 = 1.552e-6

colorA = ['blue', 'black', 'red']

f, ax = plt.subplots(1, 1)
f1, ax1 = plt.subplots(1, 1)

# Loop over different positions in the laser
for i in range(3):
    # Load the data
    dataA = np.loadtxt('output/output_redo_full.txt', dtype=complex, usecols=i)

    # Plot the pulse shape
    [valid_flag, t_axis, trace_selection, trace_selection_fit, pulsewidth, pulseenergy] = getpulse(timeStep, dataA, 1e9)
    maxNdx = np.argmax(np.abs(trace_selection)**2)
    maxValue = np.max(np.abs(trace_selection)**2)
    translated_t_axis = t_axis - t_axis[maxNdx]

    ax.plot(translated_t_axis, np.abs(trace_selection)**2 / maxValue, label='Position ' + str(i + 1), c=colorA[i])

    # Plot the spectrum envelope
    [valid, lambdas, spectrum, lambdas_envelope, envelope, lambdas_phases, phases] = getspectrum(timeStep, lambda0, dataA, get_envelope_and_phase=True)
    if valid == True:
        print('PE = ' + str(pulseenergy))
        print('PW = ' + str(pulsewidth))
        freq_center = int(np.where(spectrum == max(spectrum))[0])
        lambda_center = lambdas[freq_center]
        ax1.plot(lambdas_envelope, envelope, label='Position ' + str(i + 1), c=colorA[i])

ax.set_xlabel('$t - t_0$ [ps]')
ax.set_ylabel('Amplitude [a.u.]')
ax.legend()

ax1.set_xlabel('Wavelength [nm]')
ax1.set_ylabel('Amplitude [5 dB per div]')
yticks = np.arange(np.min(envelope), np.max(envelope) + 5, step=5)
ax1.set_yticks(yticks)
ax1.set_yticklabels([' ' for i in range(len(yticks))])
ax1.legend()

plt.show()