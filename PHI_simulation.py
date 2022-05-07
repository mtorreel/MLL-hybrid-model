import os
from shutil import copyfile
from video import Video
import numpy as np
import matplotlib.pyplot as plt

# Global functions and variables
c_l = 299792458 # speed of light

def copyAndRemoveFile(originDirectory, destinationDirectory, fileName):
            '''Copy and remove the file from an origin to a destination. Works if the os in the origin directory.'''

            # Copy output file to output directory
            copyfile(originDirectory + '/' + fileName, destinationDirectory + '/' + fileName)

            # Remove output file in input directory
            os.remove(fileName)

# Class for PHIsim simulations
class PHI_simulation():
    # Dictionary to look up all parameters in par file
    parameter_to_NdxD = {
        'gainCoeff_SA': 1,
        'epsilonTPA_SA': 18,
        'TPA_SA': 20,
        'lifeTime_SA': 23,
        'bimol_SA': 26,
        'Auger_SA': 27,
        'drift_SA': 28,
        'nrOfWL': 52,
        'FWHM_SOA': 58,
        'FWHM_SA': 59,
        'gbar_SOA': 60,
        'gbar_SA': 61
    }

    # dictionary with the default values for parameters
    parameter_defaultD = {
        'gainCoeff_SA': 1.694e-19,
        'epsilonTPA_SA': 200.0,
        'TPA_SA': 3.7e-10,
        'lifeTime_SA': 15e-12,
        'bimol_SA': 2.620e-16,
        'Auger_SA': 5.269e-41,
        'drift_SA': 5.07e-102,
        'nrOfWL': 20,
        'FWHM_SOA': 1e12,
        'FWHM_SA': 1e12,
        'gbar_SOA': 1e3,
        'gbar_SA': 1e3
    }

    def __init__(self, nrOfSpaceSteps, nrOfCycles = 0, videoSamplingRate = -1, folderDirectory = 'C:/Users/Jany/thesis_code/hybridPHISim', isWithMemory=True, current=0, isInitialized=False):
        
        # Directories
        self.folderDirectory = folderDirectory        
        self.input_work_dir = folderDirectory + '/input'
        self.output_dir = folderDirectory + '/output'

        # Change nrOfCycles if parameter is different from 0
        if nrOfCycles != 0:
            self.set_nrOfCycles(nrOfCycles)
        
       # Set the video sampling rate
        self.videoSamplingRate = videoSamplingRate
        self.set_videoSamplingRate(videoSamplingRate)
        if videoSamplingRate != -1:
            self.video = Video(folderDirectory=folderDirectory)
        
        # Save the important characteristics of the simulation
        charA = self.read_pars()
        self.centralWL = charA[0]
        self.groupNdx = charA[1]
        self.nrOfWL = charA[2]
        self.nrOfCycles = charA[3]
        self.nrOfSpaceSteps = nrOfSpaceSteps

        self.timeStep = self.nrOfWL * self.centralWL / c_l;  # this the time per segment in seconds
        self.totalTime = self.nrOfCycles * self.timeStep
        self.simulationTime = 0     # time that has been simulated

        # Set the current
        if current != 0:
            self.current = current
            self.set_current()

        # Generate the input files with initialization - does not need to be done when already initialized
        if isInitialized == False:
            self.generateInputFiles()

        # Set memory for charge and photon density on/off
        self.reset_carrierfile()
        self.reset_photondfile()
        self.isWithMemory = isWithMemory
    
    def read_pars(self):
        """ Reads and returns from the inputfile:
            central wavelength
            group index
            number of wavelengths per segment
            number of steps in the simulation
            name of the file with output data
        """  
        # Go to correct working directory
        os.chdir(self.input_work_dir)

        # we want to read from the PHIsim_invoer_amp.txt (line numbers start from 1)
        cwl_i = 32  #  the central wavelength  - line 32 in the file
        ngr_i = 45  #  the group index  - line 45 in the file
        nwl_i = 53  #  the number of wavelengths per segment  - line 53 in the file
        nst_i = 54  #  the number of steps in the simulation  - line 54 in the file
        ofn_i = 55  #  the name of the output file for PHIsim - line 55 in the file

        f_par_in = open("PHIsimv3_pars_InGaAsP_ridge.txt", 'r')     # open the file for reading
        for i in range(1,58):
            inf_line    = f_par_in.readline()        # read one line in the file
            inf_sp_line = inf_line.replace('\t',' ') # now replace all tabs with spaces
            all_line    = inf_sp_line.split(' ')     # split the string into parts (space is separator)

            if i==cwl_i:
                cwl=np.float_(all_line[0])
            if i==ngr_i:
                n_gr=np.float_(all_line[0])
            if i==nwl_i:
                n_wl_seg=np.int_(all_line[0])
            if i==nst_i:
                n_cycles=np.int_(all_line[0])
            if i==ofn_i:
                PHIoutputfile=all_line[0]

        f_par_in.close # close the file

        return(cwl, n_gr, n_wl_seg, n_cycles, PHIoutputfile)

    def set_nrOfCycles(self, nrOfCycles):
        '''Sets the number of cycles in the simulation. This has to be changed in the simulation parameters file.'''

        self.nrOfCycles = nrOfCycles

        # Go to correct working directory
        os.chdir(self.input_work_dir)

        # Open the input file
        file = open("PHIsimv3_pars_InGaAsP_ridge.txt", "r")
        replacement = ""
        
        for i, line in enumerate(file):
            if i == 53:
                replacement = replacement + str(nrOfCycles) + '	        #	Number of cycles in the simulation\n'
            else:
                replacement = replacement + line

        file.close()

        # Opening the input file in write mode
        fout = open("PHIsimv3_pars_InGaAsP_ridge.txt", "w")
        fout.write(replacement)
        fout.close()

    def set_videoSamplingRate(self, videoSamplingRate):
        '''Sets the video sampling rate. -1 means no video data is stored. This has to be changed in the simulation parameters file.'''

        # Go to correct working directory
        os.chdir(self.input_work_dir)

        # Open the input file
        file = open("PHIsimv3_pars_InGaAsP_ridge.txt", "r")
        replacement = ""
        
        for i, line in enumerate(file):
            if i == 55:
                replacement = replacement + str(videoSamplingRate) + '	        #	if > 0 output data for video generated, store data for video every N time\n'
            else:
                replacement = replacement + line

        file.close()

        # Opening the input file in write mode
        fout = open("PHIsimv3_pars_InGaAsP_ridge.txt", "w")
        fout.write(replacement)
        fout.close()

    def set_otherParameters(self, parNameA, parValueA):
        '''Sets other parameters whose parameter-index pairs have been defined in the global dictionary.'''

        # Go to correct working directory
        os.chdir(self.input_work_dir)

        # Open the input file
        file = open("PHIsimv3_pars_InGaAsP_ridge.txt", "r")
        replacement = ""

        # Get the index for the parameter
        parNdxA = []
        for parName in parNameA:
            parNdxA.append(self.parameter_to_NdxD[parName])
        
        for i, line in enumerate(file):
            if i in parNdxA:
                index = parNdxA.index(i)
                replacement = replacement + str(parValueA[index]) + '\n'
            else:
                replacement = replacement + line

        file.close()

        # Opening the input file in write mode
        fout = open("PHIsimv3_pars_InGaAsP_ridge.txt", "w")
        fout.write(replacement)
        fout.close()

    def reset_otherParameters(self):
        '''Resets the other parameters to the default values as enlisted in parameter_defeaultD.'''
        for key in self.parameter_to_NdxD.keys():
            self.set_otherParameters([key], [self.parameter_defaultD[key]])

    def reinitialize_importantParameters(self):
        # Save the important characteristics of the simulation
        charA = self.read_pars()
        self.centralWL = charA[0]
        self.groupNdx = charA[1]
        self.nrOfWL = charA[2]
        self.nrOfCycles = charA[3]
        self.nrOfSpaceSteps = self.nrOfSpaceSteps

        self.timeStep = self.nrOfWL * self.centralWL / c_l;  # this the time per segment in seconds
        self.totalTime = self.nrOfCycles * self.timeStep
        self.simulationTime = 0  # time that has been simulated

    def set_current(self):
        '''Sets the current flowing to the SOAs. This has to be changed in the simulation parameters file.'''

        # Go to correct working directory
        os.chdir(self.input_work_dir)

        # Open the input file
        file = open("device_input.txt", "r")
        replacement = ""
        
        for i, line in enumerate(file):
            if i == 13:
                replacement = replacement + '0     ' + str(np.round(self.current * 0.2, decimals=3)) + '\n'
            elif i == 14:
                replacement = replacement + '1     ' + str(np.round(self.current * 0.8, decimals=3)) + '\n'
            else:
                replacement = replacement + line

        file.close()

        # Opening the input file in write mode
        fout = open("device_input.txt", "w")
        fout.write(replacement)
        fout.close()
    
    def generateInputFiles(self):
        '''Generates the input files.
        '''

        # Go to correct working directory
        os.chdir(self.input_work_dir)

        # Invoke PHIsim_inputv3.exe
        comm_str = 'PHIsim_inputv3.exe PHIsimv3_pars_InGaAsP_ridge.txt device_input.txt'
        os.system(comm_str)

    def generateSchematic(self):
        '''Generates the network schematic. To be found in the input folder.
        '''
         # Go to correct working directory
        os.chdir(self.input_work_dir)

        # Invoke graphviz
        comm_str = 'dot -Tpng PHIinput_graph.gv -o PHIinput_graph.png'
        os.system(comm_str)

    def generateSignalInputFile_pulseInjection(self, pulse_height = 1.100, pulse_width = 3.0e-12, pulse_center = [20e-12]):
        '''Generates a signal input file for the pulse injection.'''

        # Move to correct directory
        os.chdir(self.input_work_dir)

        # Generate array with timestamps
        timesc_in=np.linspace(0, self.totalTime, self.nrOfCycles)  # create array with time stamps for the input

        # Now the arrays with data that have to written to the signals input file
        # can be defined and filled. User input required here.
        P_LR_in = np.zeros_like(timesc_in, dtype=np.float_)  # first column is the power from left to right units(W)
        F_LR_in = np.zeros_like(timesc_in, dtype=np.float_)  # second column is the phase going from left to right
        P_RL_in = np.zeros_like(timesc_in, dtype=np.float_)  # third column is the (low!) power from right to left
        F_RL_in = np.zeros_like(timesc_in, dtype=np.float_)  # fourth column is the phase going from right to left

        inputSignal = pulse_height * 1.0e-9

        for center in pulse_center:
            inputSignal += pulse_height * np.exp(-((timesc_in - center) / 2 / pulse_width)**2)
        
        P_LR_in = np.abs(inputSignal)
        F_LR_in = np.unwrap(np.angle(inputSignal))

        # Write the signal file
        signalDataM = np.zeros((len(P_LR_in), 4))
        signalDataM[:, 0] = np.transpose(P_LR_in)
        signalDataM[:, 1] = np.transpose(F_LR_in)
        signalDataM[:, 2] = np.transpose(P_RL_in)
        signalDataM[:, 3] = np.transpose(F_RL_in)

        np.savetxt('signal_input.txt', signalDataM, delimiter = ' ')

        return inputSignal

    def run(self):
        '''Run the simulation.'''
        
        # Go to correct working directory
        os.chdir(self.input_work_dir)
        
        # Run PHIsim_v3
        comm_str= 'PHIsim.exe PHIsimv3_pars_InGaAsP_ridge.txt devicefile.txt carrierfile.txt photondfile.txt signal_input.txt'
        os.system(comm_str)

        # If isWithMemory is put on: save the output carriers as carrierfile.txt and output photon density as photondfile.txt
        if self.isWithMemory == True:

            # Carrier memory
            os.remove('carrierfile.txt')
            os.rename('PHIsimout_car.txt', 'carrierfile.txt')
            # Photon density memory
            os.remove('photondfile.txt')
            os.rename('PHIsimout_opt.txt', 'photondfile.txt')
        else:
            os.remove('PHIsimout_car.txt')
            os.remove('PHIsimout_opt.txt')

        # Remove redundant files
        os.remove('PHIsimv3_log.txt')

        originDirectory = self.input_work_dir
        destinationDirectory = self.output_dir

        # Copy and remove output file
        copyAndRemoveFile(originDirectory, destinationDirectory, 'PHIsimout.txt')

        # Copy and remove the video files
        if self.videoSamplingRate != -1:
            copyAndRemoveFile(originDirectory, destinationDirectory, 'PHIsimout_vid_carriers.txt')
            copyAndRemoveFile(originDirectory, destinationDirectory, 'PHIsimout_vid_LRf.txt')
            copyAndRemoveFile(originDirectory, destinationDirectory, 'PHIsimout_vid_LRp.txt')
            copyAndRemoveFile(originDirectory, destinationDirectory, 'PHIsimout_vid_RLf.txt')
            copyAndRemoveFile(originDirectory, destinationDirectory, 'PHIsimout_vid_RLp.txt')

        # Update the simulation time
        self.simulationTime += self.totalTime
    
    def get_outputLR(self):
        '''Returns the LR output in a numpy array (power in W).'''

        # Go to correct working directory
        os.chdir(self.output_dir)

        # Load the relevant data
        dataM = np.loadtxt('PHIsimout.txt', delimiter=' ')

        return np.sqrt(dataM[:, 0]) * np.exp(1j * dataM[:, 1])

    def get_outputRL(self):
        '''Returns the RL output in a numpy array (power in W).'''

        # Go to correct working directory
        os.chdir(self.output_dir)

        # Load the relevant data
        dataM = np.loadtxt('PHIsimout.txt', delimiter=' ')

        return np.sqrt(dataM[:, 2]) * np.exp(1j * dataM[:, 3])

    def generateSignalInputFile_external(self, newInput):
        '''Generate a signal input file based on external data.'''

        # Go to correct working directory
        os.chdir(self.input_work_dir)

        # Now the arrays with data that have to written to the signals input file
        # can be defined and filled. User input required here.
        P_RL_in = np.abs(newInput) ** 2        
        F_RL_in = np.angle(newInput) # second column is the phase going from left to right

        P_LR_in = np.zeros_like(P_RL_in, dtype=np.float_)  # first column is the power from left to right units(W)
        F_LR_in = np.zeros_like(P_RL_in, dtype=np.float_)  # fourth column is the phase going from right to left

        # Write the signal file
        signalDataM = np.zeros((len(P_LR_in), 4))
        signalDataM[:, 0] = np.transpose(P_LR_in)
        signalDataM[:, 1] = np.transpose(F_LR_in)
        signalDataM[:, 2] = np.transpose(P_RL_in)
        signalDataM[:, 3] = np.transpose(F_RL_in)

        np.savetxt('signal_input.txt', signalDataM, delimiter = ' ')

    def reset_carrierfile(self):
        '''Resets the carrier file.'''

        # Go to correct working directory
        os.chdir(self.input_work_dir)

        # Remove existing carrier file
        os.remove('carrierfile.txt')

        # Write new carrier file
        f = open('carrierfile.txt', 'w')
        f.write('-1 -1 \n')
        f.close()

    def reset_photondfile(self):
        '''Resets the photon density file.'''

        # Go to correct working directory
        os.chdir(self.input_work_dir)

        # Remove existing carrier file
        os.remove('photondfile.txt')

        # Write new carrier file
        f = open('photondfile.txt', 'w')
        f.write('-1 -1 -1 -1 -1 -1 -1 \n')
        f.close()