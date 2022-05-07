import numpy as np
from scipy.optimize import curve_fit
import matplotlib.cm
import matplotlib.pyplot as plt


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"


    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    """
    print(x)
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def find_localmaxima(x, mph):
    # x: input array
    # mph: miminal peak height
    # returns local maxima, defined as points larger than mph and larger than its neighboring elements
    return np.where(np.logical_and(x > mph, np.logical_and(x - np.roll(x, 1) > 0, np.abs(x) - np.roll(x, -1) > 0)))[0]


def sech_functiondef(x, P0, x1, x2):
    # hyperbolic sech^2 function
    return P0 * ((2 * np.exp(-np.abs(x1 - x) / x2) / (1 + np.exp(-2 * np.abs(x1 - x) / x2))) ** 2)


def getspectrum(dt, lambda0, OUTPUTmonitor, get_envelope_and_phase=False):
    # Function to fourier transform pulse train and get spectrum
    # ----------------------------------------------------------------------------------------------------------
    # Arguments:
    #   dt: discretization step size [s]
    #   lambda0: central wavelength [m]
    #   OUTPUTmonitor: time trace of signal samples
    #   get_envolope_and_phase: option in case the envelope and spectral phase is to be extracted
    # Returns
    #   valid boolean (for reliability purposes, indicates whether the trace was sufficiently long)
    #   lambdas, spectrum
    #   lambdas_envelope,envelope,lambdas_envelope,phases in case get_envelope_and_phase==True

    c = 3 * 1e8  # speed of light

    outputmonitor_secondhalf = OUTPUTmonitor[int(np.round(0.5 * len(OUTPUTmonitor))):int(len(OUTPUTmonitor))]
    minpeakheight = 0.6 * np.max(np.abs(outputmonitor_secondhalf)) # a bit of an arbitrary choice...
    peakpositions = find_localmaxima(np.abs(outputmonitor_secondhalf), minpeakheight)

    n_o_peaks = len(peakpositions)
    if n_o_peaks < 20:
        print('The signal time trace is too short to reliably plot the spectrum.')
        return [False, 0, 0, 0, 0, 0, 0]

    else:
        valid = True
        pos2 = int(np.round(0.5 * (peakpositions[n_o_peaks - 1] + peakpositions[n_o_peaks - 2])))
        pos1 = int(np.round(0.5 * (peakpositions[int(np.round(0.5 * n_o_peaks)) - 1] + peakpositions[int(np.round(0.5 * n_o_peaks)) - 2])))
        trace_selection = outputmonitor_secondhalf[pos1:pos2]

        V = np.linspace(-0.5 * 2 * np.pi / dt, 0.5 * 2 * np.pi / dt, int(len(trace_selection)))
        W = V + 2 * np.pi * c / lambda0
        f = np.multiply(W, (1e-12) / (2 * np.pi))
        # spectrum = np.multiply((dt / int(len(V))), np.square(np.abs(np.fft.fftshift(np.fft.fft(trace_selection)))))
        spectrum = np.multiply((dt / int(len(V))), np.square(np.abs(np.fft.fftshift(np.fft.fft(trace_selection)))))

        if get_envelope_and_phase == True:
            envelope_n_o_points = 1000
            spectrum_envelope = np.zeros((envelope_n_o_points))
            f_envelope = np.zeros((envelope_n_o_points))
            phases = np.zeros((envelope_n_o_points))
            n_o_samp = int(np.floor(int(len(spectrum)) / envelope_n_o_points))
            for i in range(envelope_n_o_points):
                if i == len(spectrum) - 1:
                    temp_array = 10 * np.log10((1 / (1e-3)) * spectrum[1 + i * n_o_samp:int(len(spectrum))])
                    temp_max = np.max(temp_array)
                    pos_max = int(np.where(temp_array == temp_max)[0])
                else:
                    temp_array = 10 * np.log10((1 / (1e-3)) * spectrum[int(i * n_o_samp):int((i + 1) * n_o_samp - 1)])
                    temp_max = np.max(temp_array)
                    pos_max = int(np.where(temp_array == temp_max)[0])
                spectrum_envelope[i] = temp_max
                f_envelope[i] = f[pos_max + i * n_o_samp]
                phases[i] = np.angle(np.fft.fftshift(np.fft.fft(trace_selection))[pos_max + i * n_o_samp])

            lambdas = np.multiply((1e-3) * c, np.reciprocal(f))
            spectrum = 10 * np.log10((1 / (1e-3)) * spectrum)
            lambdas_envelope = (1e-3) * np.multiply(c, np.reciprocal(f_envelope))
            envelope = spectrum_envelope

            return [valid, lambdas, spectrum, lambdas_envelope, envelope, lambdas_envelope, phases]
        else:
            lambdas = np.multiply((1e-3) * c, np.reciprocal(f))
            spectrum = 10 * np.log10((1 / (1.0e-3)) * spectrum)
            return [valid, lambdas, spectrum, 0, 0, 0, 0]


def getpulse(dt, OUTPUTmonitor,estimated_rep_rate):
    # extract pulse from output monitor, get pulse width, pulse energy, etc.

    outputmonitor_secondhalf = OUTPUTmonitor[int(np.round(0.5 * len(OUTPUTmonitor))):int(len(OUTPUTmonitor))]
    minpeakheight = 0.8 * np.max(np.abs(outputmonitor_secondhalf))
    peakpositions = detect_peaks(np.abs(outputmonitor_secondhalf), mph=minpeakheight,mpd=0.1*1/(estimated_rep_rate*dt))
    n_o_peaks = len(peakpositions)
    valid_flag = True

    if n_o_peaks > 10:
        # take the second last pulse of the trace
        pos2 = int(np.round(0.5 * (peakpositions[n_o_peaks - 1] + peakpositions[n_o_peaks - 2])))
        pos1 = int(np.round(0.5 * (peakpositions[n_o_peaks - 3] + peakpositions[n_o_peaks - 2])))
        trace_selection = outputmonitor_secondhalf[pos1:pos2]

        # define time axis in picoseconds
        T_span = dt * len(trace_selection) * 1e12
        t_axis = np.linspace(-0.5 * T_span, 0.5 * T_span, len(trace_selection))

        pulsewidth=0
        pulseenergy=0
        trace_selection_fit=trace_selection
        # sech fit to pulse
        if len(trace_selection)>50: # arbitrary minimal trace length for reliable fitting
            xdata = np.linspace(-0.5 * T_span, 0.5 * T_span, len(trace_selection))
            ydata = np.square(np.abs(trace_selection))
            guess = np.array([max(ydata), 0, 1 / 1.763])
            [fit, pcov] = curve_fit(sech_functiondef, xdata, ydata, p0=guess)
            trace_selection_fit = sech_functiondef(xdata, fit[0], fit[1], fit[2])
            pulsewidth = fit[2] * 1.763  # FWHM pulse width for sech [ps]
            pulseenergy = pulsewidth * fit[0] / 0.88  # pulse energy for sech [pJ]

        return [valid_flag, t_axis, trace_selection, trace_selection_fit, pulsewidth, pulseenergy]

    else:
        valid_flag = False
        return [valid_flag, 0, 0, 0, 0, 0, 0]


def carrier_init(x, I, e, V, Nt, A, B, C):
    # carrier equation to initialize carriers at startup given the injection current I, active volume V, transparency carrier density Nt,
    # ... and the recombination parameters A,B,C
    return (I / (e * V * Nt)) - A * x - (B) * (x ** 2) - (C) * (x * x * x)


def generateHeatmaps(var1A, varA, var1Key, varKey, var1Axis, varAxis, bad_valueA):
    '''Generates the heatmaps displaying both the pulse width and energy for a 2D parameter sweep.'''
    widthA = []
    energyA = []

    for var in varA:
        # f,ax =plt.subplots(1, 1)
        for var1 in var1A:
            print('New iteration')
            if (var, var1) not in bad_valueA:
                # Load data
                outputName = 'output_' + varKey + '_' + str(var).replace('.', '_') + '_' + var1Key + '_' + str(var1).replace('.', '_') + '_withSSF'
                dataA = np.loadtxt('output/' + outputName + '.txt', dtype=complex)

                # Get pulse width and energy
                [valid_flag, t_axis, trace_selection, trace_selection_fit, pulsewidth, pulseenergy] = getpulse(1.035e-13, dataA, 1e9)

                widthA.append(pulsewidth)
                energyA.append(pulseenergy)

                # plt.plot(t_axis, np.abs(trace_selection)**2)
                # plt.plot(t_axis, trace_selection_fit)
                # plt.legend(['Pulse', 'Fit'])
                # plt.show()
            else:
                widthA.append(0)
                energyA.append(0)

        # ax.legend()
        # ax.set_xlabel('Time [s]')
        # ax.set_ylabel('Power [W]')
        # ax.set_title(varName + ' %.2e 1/m'%(var))
        # plt.show()

    # Generate heat plot for pulse width
    widthM = np.array(widthA).reshape((len(varA), len(var1A)))
    minWidth = np.trim_zeros(np.sort(widthM, axis=None))[0]

    value = 0

    masked_array = np.ma.masked_where(widthM == value, widthM)

    cmap = matplotlib.cm.Blues_r  # Can be any colormap that you want after the cm
    cmap.set_bad(color='black')

    f, ax = plt.subplots(1, 1)
    im = ax.imshow(np.transpose(masked_array)[::-1], cmap=cmap, vmin=minWidth, extent=[np.min(varA), np.max(varA), np.min(var1A), np.max(var1A)])
    f.colorbar(im, label='Pulse width [ps]')
    ax.set_xlabel(varAxis)
    ax.set_ylabel(var1Axis)
    ax.set_aspect(np.abs(varA[1] - varA[0]) / np.abs(var1A[1] - var1A[0]))
    plt.show()

    # Generate heat plot for pulse energy
    energyM = np.array(energyA).reshape((len(varA), len(var1A)))
    minEnergy = np.trim_zeros(np.sort(energyM, axis=None))[0]

    value = 0

    masked_array = np.ma.masked_where(energyM == value, energyM)

    cmap = matplotlib.cm.Blues_r  # Can be any colormap that you want after the cm
    cmap.set_bad(color='black')

    f, ax = plt.subplots(1, 1)
    im = ax.imshow(np.transpose(masked_array)[::-1], cmap=cmap, vmin=minEnergy, extent=[np.min(varA), np.max(varA), np.min(var1A), np.max(var1A)])
    f.colorbar(im, label='Pulse energy [pJ]')
    ax.set_xlabel(varAxis)
    ax.set_ylabel(var1Axis)
    ax.set_aspect(np.abs(varA[1] - varA[0]) / np.abs(var1A[1] - var1A[0]))
    plt.show()