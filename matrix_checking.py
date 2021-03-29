#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:13:19 2019

[1] Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: 
uses and interpretations. Neuroimage, 52(3), 1059-1069.
https://github.com/tsalo/brainconn

[2] Muldoon, S. F., Bridgeford, E. W., & Bassett, D. S. (2016). 
Small-world propensity and weighted brain networks. 
Scientific reports, 6, 22057.

@author: Carlos Coronel
"""

import numpy as np
import graph_utils
#import networkx as nx

def fast_checking(A, tau = 0.5, reps = 100, gamma = 1):
    """
    Compute basic metrics for characterizing a graph.
    Integration metrics: characteristic path (charpath) and
    global efficiency (global_eff).
    Segregation metrics: trans (transitivity)
    and local efficiency (local_eff).
    Community metrics: mean participation coefficient (BA) and
    modularity (Q).
    Inputs
    ======
    A: array, the weighted and undirected adjacency matrix for the given network.
    tau: agreement matrix absolute threshold.
    reps: number of iteration of Louvain's algorithm.
    gamma: resolution parameter of modules detection. Higher values allow the
    detection of smaller modules.
    Outputs
    =======
    metrics: array, graph measures in the next order -> charpath, global_eff, BA,
    trans, local_eff, Q
    """
 
    D_matrix = graph_utils.get_agreement_matrix(A, reps, tau, gamma) #Agreement matrix
    consensus_vector = graph_utils.consensus_und(D_matrix, tau, reps) #consensus vector 
    
    #Check integration
    global_eff = graph_utils.efficiency_wei(A, local = False)
    distance = graph_utils.distance_wei(graph_utils.invert(A))[0]
    charpath = graph_utils.charpath(distance, include_diagonal = False,
                                    include_infinite = True)[0]
    #Check segregation
    local_eff = np.mean(graph_utils.efficiency_wei(A, local = True))
    trans = graph_utils.transitivity_wu(A)        
    
    #Community and individual metrics
    BA = np.mean(graph_utils.participation_coef(W = A, ci = consensus_vector, degree = 'undirected'))
    Q = graph_utils.modularity_und(A = A, kci = consensus_vector, gamma = gamma)[1]
 
    return(np.array([charpath,global_eff,BA,trans,local_eff,Q]))
