# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:27:32 2021

@author: Carlos Coronel
"""

import numpy as np
from scipy import stats
import graph_utils
from Network_Surrogates import DSPR

#Load the structural connectivity
M = np.loadtxt('structural_Deco_AAL.txt')
nnodes = len(M)


#%%
###LOCAL METRICS###

#Distance matrix 
distance = graph_utils.distance_wei(graph_utils.invert(M))[0]
#Nodal efficiency
nodal_eff = [1/np.mean(distance[x,list(range(0,90)).remove(x)]) for x in range(0,90)]
#Nodes' degree
degree = np.sum(M > 0,0)
#Nodes' strength
strength = np.sum(M,0)
#Clustering coefficient
clus_coeff = graph_utils.clustering_coef_wu(M)
 
#%%
###RICH CLUB ORGANIZATION (MESO-SCALE)###

CM = np.copy(M) #Copy of M
#This calculates the rich club coefficient of CM for each k value (k = degree)
rich_club, all_non_rich_nodes = graph_utils.rich_club_wu(CM)

#vector of k values
k_vector = np.arange(0,len(rich_club),1)   
iters = 1000 #number of randomizations of CM
#rich club coefficient for each random matrix and each w value
rich_rand_vec = np.zeros((iters,len(rich_club)))
for i in range(iters):
    np.random.seed(i)
    random_mat = DSPR(CM)    
    #Rich club coefficient (w x iters)
    rich_rand_vec[i,:] = graph_utils.rich_club_wu(random_mat,len(rich_club))[0]

 
#mean value across iterations
rich_random =  np.nanmean(rich_rand_vec,0)
#standard deviation across iterations
rich_rand_std = np.nanstd(rich_rand_vec,0)
#Vector for saving p-values
p_vals = np.zeros(len(rich_club))

#This uses the mean (mu) and standard deviation (sigma) of random matrices, for each w value,
#to fit a normal distribution. Then, it was computed the p-value for each rich club coefficient
#("rich_club") calculated from CM.
for i in range(0,len(rich_club)):
    mu, sigma = rich_random[i], rich_rand_std[i]
    p_vals[i] = 1 - stats.norm.cdf(rich_club[i], mu, sigma)
p_vals[np.isnan(p_vals)] = 1

#Normalized rich_club coefficient
rich_club[np.isnan(rich_club)] = 0.01
rich_random[np.isnan(rich_random)] = 0.01
rich_norm = rich_club / rich_random #This is the normalized coefficient

#Maximal normalized rich club coefficient
rich_values = np.argwhere(p_vals < 0.05)
rich_max = np.argmax(rich_norm * (p_vals < 0.05))
rich_idx = rich_max


#This part identifies the nodes that are connected with the rich club nodes, that is,
#the feeders nodes. An absolute threshold was applied to CM because the original matrix, M,
#Is very dense (approx. 40%). The number of feeder nodes decreases with the threshold.

#Nodes that do not belong to the rich club
non_rich_nodes = all_non_rich_nodes[rich_max]
    
#Rich club nodes
rich_nodes = np.delete(np.arange(0,90,1),non_rich_nodes)

#In briefly, this part find which non-rich club nodes are connected to rich club members,
#using simple matrix operations
rich_mat = np.zeros((90,90))
rich_mat[rich_nodes,:] = 1
non_rich_mat = np.zeros((90,90))
non_rich_mat[:,non_rich_nodes] = 1
non_rich_mat *= (CM>0.05)
feeders_mat = rich_mat + non_rich_mat
feeders_nodes = np.argwhere(np.max(feeders_mat,0)==2)[:,0]

#Finding the local nodes
feeders_mat[:,rich_nodes] = 3
if len(non_rich_nodes) == len(feeders_nodes):
    local_nodes = []
else:
    local_nodes = np.argwhere(np.max(feeders_mat,0)<2)[:,0]


#Print useful info
print('N° rich club nodes = %i, N° feeders = %i, N° local = %i'
      %(len(rich_nodes),len(feeders_nodes),len(local_nodes)))
print('Normalized Rich Club Coefficient (k = %.2f) = %.3f'%(
        k_vector[rich_idx],rich_norm[rich_idx]))
print('p-value = %.3f'%p_vals[rich_idx])


#%%
###S-CORE DECOMPOSITION (MESO-SCALE)###

#Vector of strength values. Used in the s-core decomposition
s_vector = np.linspace(0.8,2,51)

#Vector of nodes with a strength < s
remaining_nodes = np.zeros(len(s_vector))
#Matrices with nodes (and their connections) with strength < s
thesholded_matrices =  []

#S-core decomposition for each s value in s_vector
for i in range(0,len(s_vector)):
    SM, ss = graph_utils.score_wu(M,s=s_vector[i])
    remaining_nodes[i] = ss  
    thesholded_matrices.append(SM)

#Find where the number of nodes changes with s
nodes_diff = np.argwhere(np.diff(remaining_nodes) != 0 )

#Vector to save the smax values
smax = np.zeros(nnodes)

#Calculate the smax for each node
for i in range(0,len(nodes_diff)):
    nodes_idx = (np.sum(thesholded_matrices[nodes_diff[-i-1][0]],0) != 0) &\
                (smax == 0)
    smax[nodes_idx] = s_vector[nodes_diff[-i-1]]


#Smax values and counts    
smax_blocks = np.unique(smax,return_counts=True)

#Defining categories for smax vals
nodes = np.arange(0,nnodes,1)
s1 = nodes[smax < 1.48] #nodes with lower smax
s2 = nodes[(smax > 1.48) & (smax < 1.54)] #intermediate nodes
s3 = nodes[smax > 1.54] #nodes with higher smax








