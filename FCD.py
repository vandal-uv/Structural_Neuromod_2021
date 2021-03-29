# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:38:37 2020

@author: Carlos Coronel

Computes the Functional Connectivity Matrix (FCD) using as input the time series
related to neuronal activity. Also calculates the FCD's speed (dtyp) and variance.

[1] Orio, P., Gatica, M., Herzog, R., Maidana, J. P., Castro, S., & Xu, K. (2018). 
Chaos versus noise as drivers of multistability in neural networks. Chaos: 
An Interdisciplinary Journal of Nonlinear Science, 28(10), 106321.

[2] Battaglia, D., Boudou, T., Hansen, E. C., Lombardo, D., Chettouf, S., Daffertshofer, 
A., ... & Jirsa, V. (2020). Dynamic Functional Connectivity between Order and 
Randomness and its Evolution across the Human Adult Lifespan. bioRxiv, 107243.
"""

import numpy as np
import numpy.linalg as LA

#Distance metrics for the FCD:
#mode = 1 -> Pearson correlation
#mode = 2 -> Clarkson distance
names = ['corr', 'clarksondist']

def extract_FCD(data, L = 50, mode = 1, dt = 1, steps = 1):
    """
    Computes the FCD matrix. First, calculates all the Functional Connectivity
    (FC) matrices over time. Next, Uses the FCs matrices to build the FCD matrix.
    
    Parameters
    ----------
    data : txN numpy array.
           time series. Rows represent the time, and columns the nodes.
    L : integer.
        Time windows' length in time units (e.g., seconds).
    mode : integer.
           1: Pearson Correlation based distance (values between 0 and 1).
           2: Clarkson distance (values between 0 and 1).
    dt : float. 
         inverse of the sampling rate in time units (e.g., seconds).
    steps : integer > 0.
            number of points to advance for calculating the next FC.
    Returns
    -------
    FCD : LxL numpy array.
          Functional Connectivity Dynamics matrix with L total time windows.
    L_points : integer.
               Total number of time windows.
    steps : integer > 0.
            The selected step.
    """
    
    nnodes = data.shape[1] #Number of nodes

    L_points = int(np.round(L / dt, 0)) #Time windows' length in points
    
    N_windows = (data.shape[0] - L_points) // steps + 1 #Total number of time windows
    
    all_corr_matrix = [] #vector to append the FCs matrices
    
    #FCs built using the Pearson Correlation 
    #and neglecting negative values.
    for i in range(0,N_windows):
        idx1, idx2 = 0 + i * steps, L_points + i *steps
        corr_matrix = np.corrcoef(data[idx1:idx2,:].T)
        np.fill_diagonal(corr_matrix, 0)
        all_corr_matrix.append(corr_matrix)
    
    #Vectorized versions of FCs matrices.
    corr_vectors = np.array([allPm[np.tril_indices(nnodes, k = -1)] for allPm in all_corr_matrix])
       
    X = np.shape(corr_vectors)[0]
    FCD = np.zeros((X,X))
     
    if mode in [1,2]:
        modeFCD = names[mode-1]
    else:
        raise ValueError('Select a valid mode for the FCD')
        
    #Computing the FCD
    if modeFCD == 'corr': #Correlation-based distance
        CV_centered=corr_vectors - np.mean(corr_vectors,-1)[:,None]
        FCD = 1 - np.abs(np.corrcoef(CV_centered))
    elif modeFCD == 'clarksondist': #Clarkson distance
        for ii in range(X):
            for jj in range(ii):
                FCD[ii,jj]= LA.norm(corr_vectors[ii,:]/LA.norm(corr_vectors[ii,:]) - corr_vectors[jj,:]/LA.norm(corr_vectors[jj,:]))  
                FCD[jj,ii]=FCD[ii,jj]
        FCD /= np.sqrt(2)
   
    return(FCD, L_points, steps)


def FCD_vars(FCD, L_points, steps, bins = 20, vmin = 0, vmax = 1):
    """
    Calculates the FCD's speed (dtyp) and FCD's variance (varFCD). The typical
    FCD speed corresponds to the median of the histogram of the FCD values, 
    through the diagonal of the FCD with a L_points/steps offset. The varFCD 
    is computed as the variance of FCD values of the upper triangle of the FCD 
    matrix (using the same offset). 
   
    Parameters
    ----------
    FCD : LxL numpy array.
          Functional Connectivity Dynamics matrix with L total time windows.
    L_points: integer.
              Total number of time windows.
    steps : integer > 0.
            number of time points used to advance between consecutives FCs.
    bins : integer > 0.
           number of bins (intervals) for building the histogram of FCD values.
    vmin, vmax : float, vmax > vmin.
                 Limits of the histogram.
    Returns
    -------
    dtyp : float.
           Typical FCD speed.
    varFCD : float.
             variance of the FCD.
    """
    
    
    offset = int(L_points / steps) #FCD values away from the diagonal
    
    distance = [FCD[XY,XY + offset] for XY in range(len(FCD) - offset)]                    
    histogram = np.histogram(distance, bins, range = (vmin,vmax))
    dtyp = histogram[1][np.argmax(histogram[0])] #FCD speed
    
    varFCD = np.var(FCD[np.triu_indices(FCD.shape[0], k = offset)]) #FCD variance
    
    return([dtyp,varFCD])
    


