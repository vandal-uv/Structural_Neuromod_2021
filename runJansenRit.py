# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:57:25 2018

@author: Carlos Coronel

Modified version of the Jansen and Rit Neural Mass Model [1,2].
The script runs the model for generate EEG-like and BOLD-like signals.

All the graph metrics employed were taken from the Brain Connectivity Toolbox for Python [4]:
https://github.com/aestrivex/bctpy                                                                            

[1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked 
potential generation in a mathematical model of coupled cortical columns. 
Biological cybernetics, 73(4), 357-366.

[2] Coronel-Oliveros, C., Cofré, R., & Orio, P. (2021). Cholinergic 
neuromodulation of inhibitory interneurons facilitates functional 
integration in whole-brain models. PLoS computational biology, 17(2), e1008737.

[3] Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: 
uses and interpretations. Neuroimage, 52(3), 1059-1069.

"""

import numpy as np
from scipy import signal
import time
import BOLDModel as BD
import JansenRitModel as JR
import graph_utils
import FCD
import matplotlib.pyplot as plt
import importlib
from matrix_checking import fast_checking
importlib.reload(JR)

#Simulation parameters
JR.dt = 1E-3 #Integration step
JR.teq = 60 #Simulation time for stabilizing the system
JR.tmax = 600 #Length of simulated signals
ttotal = JR.teq + JR.tmax #Total simulation time
JR.downsamp = 10 #Downsampling to reduce the number of points        
Neq = int(JR.teq / JR.dt / JR.downsamp) #Number of points to discard
Nmax = int(JR.tmax / JR.dt / JR.downsamp) #Number of points of simulated signals
Ntotal = Neq + Nmax #Total number of points of simulation

seed = 0 #Random Seed

#Network parameters
JR.M = np.loadtxt('structural_Deco_AAL.txt') #structural connectivity
JR.norm = np.mean(np.sum(JR.M,0)) #normalization
JR.nnodes = len(JR.M) #number of nodes
nnodes = JR.nnodes

#NOTE: for changing the network topology use the code "Network_Surrogates.py", and applied
#the functions over the original structural connectivity matrix. Remember re-normalize after
#the change of the structural connectivity.

#Selective neuromodulation
basal_r0 = 0.33 #Slope of pyramidal neurons sigmoid function (BASAL VALUE FOR OVERALL NODES)
nodes_subset = np.ones(nnodes) #Here specify the nodes that will be neuromodulated
target_r0 = 0.67 #Target r0 value for nodes_subset

#NOTE: nodes_subset may be replaced by the subset of nodes of interest, ordered by strength,
#efficiency, etc. In the code "NodesSubsets.py" it can be found examples using a human structural
#connectivity. Use "nodes_subset = np.ones(nnodes)" to neuromodulate all nodes simultaneously 

#Nodes parameters
JR.sigma = 2 #Input standard deviation
JR.alpha = 0 #Long-range pyramidal to pyramidal coupling
#Slope of pyramidal neurons sigmoid function
JR.r0 = basal_r0 + (target_r0 - basal_r0) * nodes_subset
JR.p = 2 #Input mean

#Simulations
JR.seed = seed
init1 = time.time()
y, time_vector = JR.Sim(verbose = True)
pyrm = JR.C2 * y[:,1] - JR.C4 * y[:,2] + JR.C * JR.alpha * y[:,3] #EEG-like output of the model
end1 = time.time()

print([end1 - init1])


#%%
#Plot EEG-like signals

plt.figure(1)
plt.clf()
plt.plot(time_vector[Neq:(Neq+10000)], pyrm[Neq:(Neq+10000),:])
plt.tight_layout()


#%%
#This part calculates several measures over the EEG-like signals: global phase synchronization,
#frequency of oscillation, signal to noise ratio, and regularity.

init2 = time.time()

#Welch method to stimate power spectal density (PSD)
#Remember: dt = original integration step, dws = downsampling           
window_length = 20 #in seconds
PSD_window = int(window_length / JR.dt / JR.downsamp) #Welch time window
PSD = signal.welch(pyrm[Neq:,:] - np.mean(pyrm[Neq:,:], axis = 0), fs = 1 / JR.dt / JR.downsamp, 
                   nperseg = PSD_window, noverlap = PSD_window // 2, 
                   scaling = 'density', axis = 0)
freq_vector = PSD[0] #Frequency values
PSD_curves =  PSD[1] #PSD curves for each node
freq_min_point = int(1 / np.diff(freq_vector)[0]) #Frequency steps (in points)

#Position (pos) of the frequency (freqs) with max power, for freqs > 1
pos = np.argmax(PSD_curves[freq_min_point:,:], axis = 0)
freqs = freq_vector[freq_min_point:][pos]
     
#Mean and variance of the frequency for all the oscillators
Mfreq = np.mean(freqs)
Varfreq = np.var(freqs)

#This is for avoiding negative values of the minimum frequency of the filter 
if Mfreq <= 3.5:
    Mfilt = 3.5
else:
    Mfilt = Mfreq

#Filtering signals
Fmin, Fmax = Mfilt - 3, Mfilt + 3
a0, b0 = signal.bessel(3, [2 * JR.dt * Fmin * JR.downsamp, 
                           2 * JR.dt * Fmax * JR.downsamp], btype = 'bandpass')
Vfilt = signal.filtfilt(a0, b0, pyrm, axis = 0)
     
#Synchronization
phases_signal = np.angle(signal.hilbert(Vfilt[Neq:,:], axis = 0)) #Phases
phaseSynch = graph_utils.simple_order_parameter(phases_signal) #Kuramoto order parameter
meanSynch = np.mean(phaseSynch) #Averaged Kuramoto order parameter
varSynch = np.var(phaseSynch) #Variance of the Kuramoto order parameter (Metastability)

end2 = time.time()

print([end2 - init2])


#%%

#Power spectral density functions
plt.figure(3)
plt.clf()
plt.plot(freq_vector[1:-2], 10 * np.log10 (PSD_curves[1:-2,:]))
plt.tight_layout()

#Kuramoto order parameter
plt.figure(4)
plt.clf()
plt.plot(phaseSynch)
plt.ylim(0,1)
plt.tight_layout()

    
#%%
#fMRI-BOLD response
init3 = time.time()

if np.any(JR.r0 == 0):
    rE = JR.s(pyrm, JR.r0 + 1E-4)
else:
    rE = JR.s(pyrm, JR.r0)

BOLD_signals = BD.Sim(rE, nnodes, JR.dt * JR.downsamp)
BOLD_signals = BOLD_signals[Neq:,:]

BOLD_downsamp = 100
BOLD_dt = JR.dt * JR.downsamp * BOLD_downsamp
BOLD_signals = BOLD_signals[::BOLD_downsamp,:]

#Filter the BOLD-like signal between 0.01 and 0.1 Hz
Fmin, Fmax = 0.01, 0.1
a0, b0 = signal.bessel(3, [2 * BOLD_dt * Fmin, 2 * BOLD_dt * Fmax], btype = 'bandpass')
BOLDfilt = signal.filtfilt(a0, b0, BOLD_signals[:,:], axis = 0)

#Surrogate thresholding
#sFC: static Functional Connectivity (sFC) matrix
sFC_BOLD = graph_utils.probabilistic_thresholding(BOLDfilt, 500, 0.05)[0]

end3 = time.time()

print([end3 - init3])


#%%

#Filtered BOLD-like signals
plt.figure(5)
plt.clf()
plt.plot(BOLDfilt)
plt.tight_layout()

#sFC matrix
plt.figure(6)
plt.clf()
plt.imshow(sFC_BOLD, cmap = 'RdBu', vmin = -1, vmax = 1)
plt.tight_layout()

#%%
#Graph Analysis

init4 = time.time()

#The function below calculates several graph metrics.
#Inputs: weighted and undirected functional connectivity matrix; threshold for the agreement matrix (tau);
#Number of iterations of the Louvain algorithm (reps), resolution parameter of the Louvain algorithm (gamma)
results = fast_checking(sFC_BOLD, tau = 0.5, reps = 200, gamma = 1) 

#CP: characteristic path; GE: global efficiency; PC: mean participation coefficient
#TT: Transitivity; LE: local efficiency; QW: modularity

CP, GE, PC = results[0], results[1], results[2]
TT, LE, QW = results[3], results[4], results[5]

end4 = time.time()

print([end4 - init4])

#%%

#Functional Connectivity Dynamics (FCD)

init5 = time.time()

L = 100 #Window length (in seconds)
mode = 2 #Mode of the FCD. 1: Pearson, 2: Clarkson

FCD_matrix, L_points, steps = FCD.extract_FCD(BOLDfilt, L = L, mode = mode, dt = BOLD_dt, steps = 2)

#Calculate the typical FCD speed (dtyp) and FCD variance (varFCD, multistability)
dtyp, varFCD = FCD.FCD_vars(FCD_matrix, L_points, steps, bins = 20, vmin = 0, vmax = 1)

#Plot the FCD
plt.figure(7)
plt.clf()
plt.imshow(FCD_matrix, vmin = 0, vmax = 1, cmap = 'jet')

end5 = time.time()

print([end5 - init5])


