#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:42:52 2020

@author: Samy Castro

This script generates different types of network surrogates: DSPR [1], random, and
homogeneous networks.

[1] Rubinov, M., & Sporns, O. (2011). Weight-conserving characterization of complex 
functional brain networks. Neuroimage, 56(4), 2068-2079.

"""

import numpy as np
import copy
import graph_utils


def DSPR(CJK):
    """
    Degree- and strength- preserving randomization (DSPR) based on [1].
    The algorithm randomizes a matrix conserving the degree (perfect) and 
    strength (approximately) distributions.
    
    Parameters
    ----------   
    CJK : numpy array.
          structural connectivity to randomize.
    Returns
    -------
    CIJ2: numpy array.
          DSPR network surrogate.
    """
    
    CIJ = copy.deepcopy(CJK)
    
    #Randomize the connection weights
    nonzeroplaces    = np.where(CIJ!=0) #list of index of connections 
    nonzerovalues    = CIJ[nonzeroplaces]
    idx_permutation  = np.random.permutation(len(nonzerovalues)) #randomize index
    
    #Assign the randomized weights
    CIJ2=np.zeros_like(CIJ)
    nonzeroplaces2=np.where(CJK!=0)
    CIJ2[nonzeroplaces2]=nonzerovalues[idx_permutation]

    
    return CIJ2

def random(CJK):
    """
    Pure randomization of the structural connectivity.
    Parameters
    ----------   
    CJK : numpy array.
          structural connectivity to randomize.
    Returns
    -------
    CIJ2: numpy array.
          Random network surrogate.
    """
    
    CIJ = np.copy(CJK)
    CIJ2 = graph_utils.get_uptri(CIJ)
    np.random.shuffled(CIJ2)
    CIJ2 = graph_utils.matrix_recon(CIJ2)
    
    return CIJ2

def homogeneous(CJK,threshold):
    """
    Homogenezation of the structural connectivity.
    Parameters
    ----------   
    CJK : numpy array.
          structural connectivity to randomize.
    threshold: cutt-off value for thresholding the structural connectivity. 
    Returns
    -------
    CIJ: numpy array.
         Homogeneous network surrogate.
    """
    
    CIJ = np.copy(CJK)
    CIJ[CIJ < threshold] = 0
    CIJ[CIJ > 0] = 1
    
    return CIJ
    






