import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Global variables and functions

c = 299792458  # speed of light[m / s]
h = 6.626070040 * 1e-34  # planck's constant [J*s]
e = 1.60217662 * 1e-19  # electron charge[C]

class SSF:
    # Passive waveguide model, used for modeling passive sections of the mode-locked laser
    # Can model the passive waveguide with delay+loss (faster, less accurate)
    # Alternatively, the passive waveguide can be modeled through a split-step Fourier method to incorporate
    # ... waveguide dispersion, nonlinearity, etc.
    # ----------------------------------------------------------------------------------------------------------
    # Arguments required for initialization of the TWM class:
    # dt: discretization step size [s]
    # lambda: central wavelength of operation [m]
    # lp: length of the passive waveguide [m]
    # cum_delay_other_components: the reservoir saves the output samples of the TWM for split-step Fourier propation
    #                 ... the size of the reservoir should resemble the entire mode-locked laser roundtrip time (to enable the split-step Fourier method to be used consistently)
    # 				  ... (note this is in contrast to the first hybrid model demonstration of https://doi.org/https://doi.org/10.1038/s41598-021-89508-6),
    #                 ... therefore, this class should know the delay of the remaining mode-locked laser building blocks [in units of seconds]
    #                 ... this parameter is only critical when SSF is used, otherwise it is not used 
    # SSF (boolean): enable/disable split-step Fourier propagation for simulating the passive waveguide

    def __init__(self, dt, lambda0, lp, cum_delay_other_components, Parameters=None):
        self.dt = dt  # step time [s]
        self.lambda0 = lambda0  # wavelength [m]
        self.lp = lp  # length of passive waveguide [m]
        self.SSF = SSF # enable/disable split-step Fourier propagation (default is disable)

        # -- Physical constants --
        self.c = c
        self.h = h
        self.e = e

        # -- Waveguide parameters --
        if Parameters==None: # default parameters
            self.n_g = 3.85  # group index
            self.neff = 3  # effective mode index in SOA/SA
            self.loss_passive = 0.7 * 1e1  # time-independent loss [dB/m]
            self.beta2 = 1.3e-24  # [s^2/m], positive means normal dispersion, negative anomalous
            self.beta3 = 0.0042e-36  # [s^3/m], third  order dispersion
            self.n2 = 5 * 1e-18  # nonlinear coefficient n2 [m^2/W] (according to PhD Bart:  n2=6e-18 m^2/W)
            self.Aeff = 0.29 * 1e-12  # effective mode area [m^2]

            # parameters for FCA, TPA
            self.btpa = 6e-12 # two-photon absorption parameter [m/W]
            self.kc = 1.35e-27 # free carrier dispersion [m^3]
            self.sigma = 1.45e-21 # free carrier absorption [m^2]
            self.tau = 1e-9 # free carrier lifetime [s]

        else:
            self.n_g = Parameters[0]  # group index
            self.neff = Parameters[1]  # effective mode index in SOA/SA
            self.loss_passive = Parameters[2]  # time-independent loss [dB/m]
            self.beta2 = Parameters[3]  # [s^2/m], positive means normal dispersion, negative anomalous
            self.beta3 = Parameters[4]  # [s^3/m], third  order dispersion
            self.n2 = Parameters[5]  # nonlinear coefficient n2 [m^2/W] (according to PhD Bart:  n2=6e-18 m^2/W)
            self.Aeff = Parameters[6]  # effective mode area [m^2]

            # parameters for FCA, TPA
            self.btpa = Parameters[7]  # two-photon absorption parameter [m/W]
            self.kc = Parameters[8]  # free carrier dispersion [m^3]
            self.sigma = Parameters[9]  # free carrier absorption [m^2]
            self.tau = Parameters[10]  # free carrier lifetime [s]

        self.s = 1  # slow light factor (set to zero for now, but can be relevant, see for example: doi:10.1038/lsa.2017.8)
        self.Nc_avg = 0  # averaged free carrier density

        # -- Dependent parameters --
        self.omega0 = 2 * np.pi * self.c / self.lambda0
        self.v_g = self.c / self.n_g # group velocity [m/s]
        self.dz = self.dt
        self.hv = self.h * self.c / self.lambda0
        self.D=(self.s**2)*2*np.pi*self.btpa/(2*h*self.omega0*(self.Aeff**2))

        k0 = 2 * np.pi / self.lambda0
        self.gamma = k0 * self.n2 / self.Aeff + 1j*(self.s**2)*self.btpa/(2*self.Aeff) # nonlinear parameter

        # Reservoir/queue are arrays holding the field samples that go in/out the passive waveguide
        # Note: the reservoir/queue arrays carry the signal samples in units of Watt [W]

        self.coldcavityRTT = ((2.0 * self.lp / self.v_g)+cum_delay_other_components) # cold cavity roundtrip time [s]
        # margin to allow exact matching of the pulse roundtrip time:
        # ... the actual pulse repetition rate can deviate slightly from the cold cavity roundtrip time due to gain/absorption, third-order dispersion
        # ... to enable exact matching of the pulse roundtrip time (necessary for SSF algorithm), take margin on the cold cavity roundtrip time
        # ... here taken to be 1% of the cold cavity roundtrip delay
        self.Reservoir_capacity_margin = int(np.round(self.coldcavityRTT/dt) / 100)
        self.Reservoir_capacity = int(int(np.round(self.coldcavityRTT/self.dt)) + self.Reservoir_capacity_margin)
        self.Reservoir = np.zeros((2, self.Reservoir_capacity), dtype=np.complex128)
        self.Queue_capacity = int(np.round(self.lp/(self.v_g*self.dt)))     # no factor 2 because there are 2 queues
        self.Queue = np.zeros((2, self.Queue_capacity), dtype=np.complex128)

    def initialize(self, sigma=0.1e-6):
        # -- initialize queue samples with noisy complex values --
        # self.Queue[:][:] = sigma * (np.random.rand(2, int(self.Queue_capacity)) + 1j * np.random.rand(2, int(self.Queue_capacity)))
        # self.Reservoir[:][:] = sigma * (np.random.rand(2, int(self.Reservoir_capacity)) + 1j * np.random.rand(2, int(self.Reservoir_capacity)))

        print('No random values inserted')
    
    def performSSF(self, inputUpper, inputLower):
        '''Avoid the looping for the queue filling and feed directly all the data to the SSF.'''

        # Put new values into the reservoir
        self.Reservoir[0, self.Reservoir_capacity - self.Queue_capacity:] = inputUpper
        self.Reservoir[1, self.Reservoir_capacity - self.Queue_capacity:] = inputLower

        # Get the starting Ndx of the SSF window
        Res_start_pos_final_1 = self.getSSFWindow(0)
        Res_start_pos_final_2 = self.getSSFWindow(1)

        # Determine time span of the SSF window
        Tspan1 = (self.Reservoir_capacity - Res_start_pos_final_1) * self.dt  # time span of window to be propagated with split-step Fourier
        Tspan2 = (self.Reservoir_capacity - Res_start_pos_final_2) * self.dt

        # Determine the phase correction arrays
        phase_corr_array1 = self.getPhaseCorrectionA(Res_start_pos_final_1, 0)
        phase_corr_array2 = self.getPhaseCorrectionA(Res_start_pos_final_2, 1)

        # Determine the propagated field
        Propagated_field_1 = np.conj(phase_corr_array1[1:len(phase_corr_array1)]) * self.Passive_Waveguide_Propagator(phase_corr_array1[1:] * self.Reservoir[0][Res_start_pos_final_1:], Tspan1)
        Propagated_field_2 = np.conj(phase_corr_array2[1:len(phase_corr_array2)]) * self.Passive_Waveguide_Propagator(phase_corr_array2[1:] * self.Reservoir[1][Res_start_pos_final_2:], Tspan2)

        # Update the queues and reservoirs
        self.Queue[0][:] = Propagated_field_1[len(Propagated_field_1) - self.Queue_capacity:]
        self.Queue[1][:] = Propagated_field_2[len(Propagated_field_2) - self.Queue_capacity:]
        self.Reservoir[:, :self.Reservoir_capacity - self.Queue_capacity] = self.Reservoir[:, self.Queue_capacity:]
        self.Reservoir[:, self.Reservoir_capacity - self.Queue_capacity:] = np.zeros((2, self.Queue_capacity))

    def Passive_Waveguide_Propagator(self, signal_in, t_span):
        # -- Split-Step Fourier propagator --
        # ARGUMENTS:
        #   - signal_in: field envelope trace for split-step Fourier propagation [units of sqrt(W)]
        #   - t_span: time span of signal_in [s]
        # OUTPUT: propagated signal trace (signal_out)
        # Note: For now, Raman effect is ignored (typically has a minor effect anyway)

        # For details on how the split-step Fourier method works: see book Nonliner fiber optics, Agrawal

        # -- parameters --
        stepsize = 200e-6  # spatial stepsize for split-step Fourier propagation [m], little bit arbitrary choice, often ~100µm is used in literature
        npas = int(np.round(self.lp / stepsize))
        alpha = np.log(10 ** (self.loss_passive / 10))
        fR = 0  # NOT USED NOW Raman contribution, see 'Nonlinear optical phenomena in silicon waveguides: modeling and applications'

        nTime = len(signal_in)
        T = np.linspace(-t_span * 0.5, t_span * 0.5, nTime)  # time grid
        dT = t_span / nTime  # time grid spacing [s]
        V = 2 * np.pi * np.linspace(-nTime * 0.5 / t_span, (nTime * 0.5 - 1) / t_span, nTime)  # angular frequency grid around omega0 [rad/s]

        # calculate average free carrier density (this is a coarse approximation of the actual loss induced by free carrier absorption, improvement of accuracy possible here)
        T_span = self.dt*(len(signal_in)-1)
        E4_avg = np.sum(np.power(np.abs(signal_in),4))/len(signal_in)
        self.Nc_avg = self.Nc_avg + T_span * (self.D*E4_avg - self.Nc_avg/self.tau)

        # Define propagation constant
        # dispersion vector (incorporates second-order, third-order, .. dispersion)
        self.betas_passive = [self.beta2, self.beta3]
        B = 0
        for i in range(1, len(self.betas_passive)):
            B = B + (self.betas_passive[i - 1] / np.math.factorial(i + 1)) * (V ** (i + 1))  # shouldn't there be an additional (-1)**(i + 1) here?
        Dispersion_operator = np.fft.fftshift(1j * B - 0.5 * alpha - 0.5*self.s*self.sigma*self.Nc_avg - 1j*self.s*self.kc*(2*np.pi/self.lambda0)*self.Nc_avg)
        operatorExpHalf = np.exp(stepsize * Dispersion_operator * 0.5)
        operatorExpFull = np.exp(stepsize * Dispersion_operator * 1.0)

        # step-wise propagation
        A = signal_in

        # First half step
        A = np.fft.ifft(operatorExpHalf * np.fft.fft(A))

        for i in range(1, npas - 1):

            # ... then nonlinearity: Kerr effect operator + Raman contribution (not implemented for now)
            K = 1j * self.gamma * ((1 - fR) * (np.abs(A) ** 2))  # + fR * nTime * dT * np.fft(np.ifft(np.fftshift(hR))*np.ifft(np.abs(A)**2)))
            A = np.exp(stepsize * K) * A

            # ... and again dispersion + losses.
            A = np.fft.ifft(operatorExpFull * np.fft.fft(A))
        
        # Final half step
        K = 1j * self.gamma * ((1 - fR) * (np.abs(A) ** 2))  # + fR * nTime * dT * np.fft(np.ifft(np.fftshift(hR))*np.ifft(np.abs(A)**2)))
        A = np.exp(stepsize * K) * A

        A = np.fft.ifft(operatorExpHalf * np.fft.fft(A))

        signal_out = A

        return signal_out
    
    def getSSFWindow(self, Ndx):
        '''Returns the starting position of the reservoir window for the given index Ndx.'''

        Res_start_pos = self.Reservoir_capacity_margin
        Res_start_pos_final = Res_start_pos

        Res_start_mag = np.abs(self.Reservoir[Ndx][Res_start_pos])
        Res_start_mag_target = np.abs(self.Reservoir[Ndx][-1])

        devi_mag = np.abs(Res_start_mag_target - Res_start_mag)

        deriv_mag =  np.gradient(np.abs(self.Reservoir[Ndx][:])) # gradient vector of the magnitude of the Reservoir samples

        deriv_mag_target = deriv_mag[-1]
        deriv_mag_start = deriv_mag[Res_start_pos]

        for jj in range(1,self.Reservoir_capacity_margin):
            if np.abs(np.abs(self.Reservoir[Ndx][Res_start_pos-jj])-Res_start_mag_target) < devi_mag:
                if np.sign(deriv_mag[Res_start_pos-jj]) == np.sign(deriv_mag_target):
                    Res_start_pos_final =  Res_start_pos-jj
                    Res_start_mag = np.abs(self.Reservoir[Ndx][Res_start_pos-jj])
                    devi_mag = np.abs(np.abs(self.Reservoir[Ndx][Res_start_pos-jj])-Res_start_mag_target)
            if np.abs(np.abs(self.Reservoir[Ndx][Res_start_pos+jj])-Res_start_mag_target) < devi_mag:
                if np.sign(deriv_mag[Res_start_pos+jj]) == np.sign(deriv_mag_target):
                    Res_start_pos_final =  Res_start_pos+jj
                    Res_start_mag = np.abs(self.Reservoir[Ndx][Res_start_pos+jj])
                    devi_mag = np.abs(np.abs(self.Reservoir[Ndx][Res_start_pos+jj])-Res_start_mag_target)

        Res_start_pos_final = Res_start_pos_final + 1
        return Res_start_pos_final

    def getPhaseCorrectionA(self, Res_start_pos_final, Ndx):
        '''Return an array to account for the phase correction for periodicity.'''

        phi1 = np.angle(self.Reservoir[Ndx][Res_start_pos_final])
        phi2 = np.angle(self.Reservoir[Ndx][self.Reservoir_capacity-1])
        phasediff = phi2 - phi1
        phase_corr_array = np.exp(1j * np.linspace(0.5 * phasediff, -0.5 * phasediff, self.Reservoir_capacity - Res_start_pos_final + 1))

        return phase_corr_array

    def performLossAndDelay(self, inputUpper, inputLower):
        '''Propagate by means of only loss and delay. In fact, here the reservoir is redundant.'''

        # Put new values into the reservoir
        self.Reservoir[0, self.Reservoir_capacity - self.Queue_capacity:] = inputUpper
        self.Reservoir[1, self.Reservoir_capacity - self.Queue_capacity:] = inputLower

        # Put the propagated values into the queue
        self.Queue[0][:] = np.exp(-self.lp * self.loss_passive * (1 / 4.343) * 0.5) * self.Reservoir[0][self.Reservoir_capacity-self.Queue_capacity:]
        self.Queue[1][:] = np.exp(-self.lp * self.loss_passive * (1 / 4.343) * 0.5) * self.Reservoir[1][self.Reservoir_capacity-self.Queue_capacity:]

        # Update reservoir values
        self.Reservoir[:, :self.Reservoir_capacity - self.Queue_capacity] = self.Reservoir[:, self.Queue_capacity:]
        self.Reservoir[:, self.Reservoir_capacity - self.Queue_capacity:] = np.zeros((2, self.Queue_capacity))

class BesselFilter:
    def __init__(self, timeStep):
        # Initialize the coefficient arrays
        self.aCoeffA = np.zeros((15,))
        self.bCoeffA = np.zeros((15,))

        self.bCoeffA[0]  = 1.073993853104e-9
        self.bCoeffA[1]  = 1.503591394345e-8
        self.bCoeffA[2]  = 9.773344063246e-8
        self.bCoeffA[3]  = 3.909337625298e-7
        self.bCoeffA[4]  = 1.075067846957e-6
        self.bCoeffA[5]  = 2.150135693914e-6
        self.bCoeffA[6]  = 3.225203540871e-6
        self.bCoeffA[7]  = 3.685946903853e-6
        self.bCoeffA[8]  = 3.225203540871e-6
        self.bCoeffA[9]  = 2.150135693914e-6
        self.bCoeffA[10] = 1.075067846957e-6
        self.bCoeffA[11] = 3.909337625298e-7
        self.bCoeffA[12] = 9.773344063246e-8
        self.bCoeffA[13] = 1.503591394345e-8
        self.bCoeffA[14] = 1.073993853104e-9

        self.aCoeffA[0]  = 1.000000000000
        self.aCoeffA[1]  = -8.793470226997
        self.aCoeffA[2]  = 36.431620104394
        self.aCoeffA[3]  = -94.163898389455
        self.aCoeffA[4]  = 169.480903575400
        self.aCoeffA[5]  = -224.527211962780
        self.aCoeffA[6]  = 225.623396810498
        self.aCoeffA[7]  = -174.592705645750
        self.aCoeffA[8]  = 104.485522542552
        self.aCoeffA[9]  = -48.097968574334
        self.aCoeffA[10] = 16.756598548956
        self.aCoeffA[11] = -4.282269732620
        self.aCoeffA[12] = 0.758553571401
        self.aCoeffA[13] = -0.083335436580
        self.aCoeffA[14] = 4.282411630876e-3

        self.timeStep = timeStep

    def apply(self, signalA):
        '''Apply the filter to the signal.'''

        zi = signal.lfilter_zi(self.bCoeffA, self.aCoeffA)
        outputSignalA, zo = signal.lfilter(self.bCoeffA, self.aCoeffA, signalA, zi=signalA[0]*zi)

        return outputSignalA


    def plotFrequencyResponse(self):

        # Retrieve the plot data
        w, h = signal.freqz(self.bCoeffA, self.aCoeffA, fs = 1 / self.timeStep)

        fig, ax1 = plt.subplots()
        ax1.set_title('Frequency Response')

        ax1.plot(w, 20 * np.log10(abs(h)), 'b')
        ax1.set_ylabel('Amplitude [dB]', color='b')
        ax1.set_xlabel('Frequency [rad/sample]')

        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g')
        ax2.grid()
        ax2.axis('tight')
        plt.show()

class CustomFilter:
    def __init__(self, b, a):
        self.b = b
        self.a = a

    def apply(self, signalA):
        '''Apply the filter to the signal.'''

        zi = signal.lfilter_zi(self.b, self.a)
        outputSignalA, zo = signal.lfilter(self.b, self.a, signalA, zi=signalA[0]*zi)

        return outputSignalA


class LorentzianFilter:
    # Optional building block: spectral filter
    # *** legacy function from hybrid model v1, not really used anymore ***
    # -- Filter class, used for convolving signal trace with a chosen filter --
    def __init__(self, dt, Lorentzian=True, MirrorBoundary=False, MirrorReflectivity=1):
        self.Lorentzian = Lorentzian
        self.dt = dt

        if Lorentzian == True:
            self.delta = 1e12  # bandwidth parameter of the Lorentzian filter
            self.filter_memory = 10 / self.delta
            self.n_o_filtersamples = int(np.round(self.filter_memory / self.dt))
            self.filter = self.delta * np.exp(-self.delta * np.linspace(0, (self.n_o_filtersamples - 1) * self.dt, self.n_o_filtersamples))
            self.filterSpectrum = np.fft.fft(self.filter)
            self.filter = np.sqrt(self.dt * np.fft.fft(np.concatenate((self.filter, np.zeros((self.n_o_filtersamples - 1))), 0)))

        else:
            print('Filter not implemented.')

    def Convolve(self, trace):
        # -- Convolution of filter with signal trace
        if self.Lorentzian == True:
            # return self.dt * sum(np.multiply(self.filter, trace))
            # evaluate filter in spectral domain
            # return self.dt * np.fft.ifft(np.multiply(np.fft.fft(np.concatenate((self.filter,np.zeros((len(trace)-1))),0)), np.fft.fft(np.concatenate((np.flip(trace), np.zeros((len(self.filter)-1))),0))))[len(self.filter)-1]
            # return np.fft.ifft(np.multiply(self.filter, np.fft.fft(np.concatenate((np.flip(trace), np.zeros((self.n_o_filtersamples - 1))), 0))))[self.n_o_filtersamples - 1]
            return np.fft.ifft(self.filter * np.fft.fft(np.concatenate((np.flip(trace), np.zeros((self.n_o_filtersamples - 1))), 0)))[self.n_o_filtersamples - 1]

        else:
            print('Filter not implemented.')
            return 0

    def plotFrequencyResponse(self):

        freqA = np.fft.fftshift(np.fft.fftfreq(len(self.filterSpectrum), self.dt))
        filterSpectrum = np.fft.fftshift(self.filterSpectrum)
        plt.loglog(freqA, filterSpectrum)
        plt.show()