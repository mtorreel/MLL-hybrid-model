from PHI_simulation import PHI_simulation, copyAndRemoveFile
from SSF import SSF, BesselFilter, CustomFilter
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

class HybridSimulation:
    def __init__(self, nrOfSpaceSteps, folderDirectory, cavityLength, simulationTime, rightReflectivity, current=0, pulse_height=0, withSSF=True, videoSamplingRate=-1, isWithMemory=True, filter=[], is_fullOutput=False):
        # Initialize PHI_simulation object
        self.phi = PHI_simulation(nrOfSpaceSteps=nrOfSpaceSteps, videoSamplingRate=videoSamplingRate, folderDirectory=folderDirectory, current=current, isWithMemory=isWithMemory)

        # Delay of other components
        cum_delay_other_components = 2 * self.phi.nrOfSpaceSteps * self.phi.timeStep

        # Initialize the SSF object
        cavityLength = cavityLength # [m]
        self.ssf = SSF(self.phi.timeStep, self.phi.centralWL, cavityLength, cum_delay_other_components)

        # Initialize the filter object
        # If a custom a and b array for the filter coefficients are used, we have to give them at initialization
        if filter == []:
            self.filter = BesselFilter(self.phi.timeStep)
        else:
            self.filter = CustomFilter(filter[0], filter[1])

        # Reflectivity of the right mirror
        self.r2 = rightReflectivity

        # Specify whether the SSF module has to be used
        self.withSSF = withSSF

        # Set the number of cycles for the PHI simulation
        self.phi.set_nrOfCycles(self.ssf.Queue_capacity)
        self.nrOfSimulations = int(simulationTime / self.phi.totalTime)
        print('Number of simulations: ' + str(self.nrOfSimulations))

        # Generate the signal input file for pulse injection
        if pulse_height == 0:
            self.phi.generateSignalInputFile_pulseInjection()
        else:
            self.phi.generateSignalInputFile_pulseInjection(pulse_height=pulse_height)

        # General features for taking account of looping
        self.simulationCounter = 0
        self.percentile = 0.1
        self.outputA = np.zeros((self.phi.nrOfCycles * self.nrOfSimulations,), dtype=complex)
        self.is_fullOutput = is_fullOutput
        if is_fullOutput == True:
            self.outputM = np.zeros((self.phi.nrOfCycles * self.nrOfSimulations, 3), dtype=complex)

        # Global timing arrays
        self.phi_timeA = []
        self.SSF_timeA = []

        # Initialize tic and toc
        self.tic = 0
        self.toc = 0
    
    def run_cycle(self):
        # Run PHI simulation
        self.phi.run()

        # Get the output
        PHI_output = self.phi.get_outputLR()

        # Filter the output signal
        # PHI_output = self.filter.apply(PHI_output)
        self.outputA[self.simulationCounter * self.phi.nrOfCycles:(self.simulationCounter + 1) * self.phi.nrOfCycles] = PHI_output
        if self.is_fullOutput == True:
            self.outputM[self.simulationCounter * self.phi.nrOfCycles:(self.simulationCounter + 1) * self.phi.nrOfCycles, 0] = PHI_output
            self.outputM[self.simulationCounter * self.phi.nrOfCycles:(self.simulationCounter + 1) * self.phi.nrOfCycles, 2] = self.phi.get_outputRL()

        self.toc = perf_counter()
        if self.simulationCounter > 1:
            self.phi_timeA.append(self.toc - self.tic)


        # Use this output as input for SSF
        self.tic = perf_counter()
        if self.withSSF == True:
            self.ssf.performSSF(PHI_output, self.ssf.Queue[0] * np.sqrt(self.r2))
        else:
            self.ssf.performLossAndDelay(PHI_output, self.ssf.Queue[0] * np.sqrt(self.r2))
        
        if self.is_fullOutput == True:
            self.outputM[self.simulationCounter * self.phi.nrOfCycles:(self.simulationCounter + 1) * self.phi.nrOfCycles, 1] = self.ssf.Queue[1]
        
        self.toc = perf_counter()
        self.SSF_timeA.append(self.toc - self.tic)

        # plt.plot(self.ssf.Reservoir[0])
        # plt.plot(self.ssf.Reservoir[1])
        # plt.show()

        self.tic = perf_counter()

        # Now generate new (filtered) input for PHI simulator from queue
        # newInput = self.filter.apply(self.ssf.Queue[1])
        # self.phi.generateSignalInputFile_external(newInput)
        self.phi.generateSignalInputFile_external(self.ssf.Queue[1])

    def run(self):
        while self.simulationCounter < self.nrOfSimulations:
            self.run_cycle()

            self.simulationCounter += 1

            # self.phi.video.play()

            if self.simulationCounter / self.nrOfSimulations > self.percentile:
                print(str(self.percentile * 100) + ' % completed')
                self.percentile += 0.1

    def saveOutput(self, fileName):
        np.savetxt(fileName, self.outputA)
        copyAndRemoveFile(self.phi.input_work_dir, self.phi.output_dir, fileName)

    def saveFullOutput(self, fileName):
        np.savetxt(fileName, self.outputM)
        copyAndRemoveFile(self.phi.input_work_dir, self.phi.output_dir, fileName)