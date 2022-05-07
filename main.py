from HybridSimulation import HybridSimulation
import numpy as np

# Function set_current() in PHI_Simulation might need to be changed when the isolator sections are inserted!!

# Inspect this!
projectName = 'structure1_v1'

# Settings
nrOfSpaceSteps = 79 + 17 + 7 # number of spatial steps in active region
folderName = 'D:/thesis_code/' + projectName # name of folder
passiveLength = 37.4e-3 # length of the passive region (m)
simulationTime = 100e-9 # total simulation time (s)
rightReflectivity = 0.99 # reflectivity of the right mirror
current = 0.080 # current fed to the amplifier sections
withSSF = True # indicates whether simulation is performed using SSF (True) or propagation and delay (False)
videoSamplingRate = -1 # rate of sampling for video (-1 means no video)
pulseHeight = 0 # pulse height - 0 means default value
isWithMemory = True # switch for photon and carrier memory
# filter = [] # coefficient arrays for a custom filter
filter = []
is_fullOutput = False    # boolean to switch for recording at multiple locations in the laser structure

print('Simulation started.')

outputName = 'output_testOutput'

# Initialize the simulation
sim = HybridSimulation(nrOfSpaceSteps, folderName, passiveLength, simulationTime, rightReflectivity, current=current, withSSF=withSSF, videoSamplingRate=videoSamplingRate, pulse_height=pulseHeight, isWithMemory=isWithMemory, filter=filter, is_fullOutput=is_fullOutput)

# Run the hybrid simulation
sim.run()

# Save the output
sim.saveOutput(outputName + '.txt')
if is_fullOutput:
    sim.saveFullOutput(outputName + '_full.txt')
print('Simulation ended.')

# Play the video
# sim.phi.video.play()