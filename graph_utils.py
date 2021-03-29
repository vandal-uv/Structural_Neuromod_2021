# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:38:25 2020

@author: Carlos Coronel

Some auxiliary functions.

All the graph metrics employed were taken from the Brain Connectivity Toolbox for Python [2]:
https://github.com/aestrivex/bctpy [2]

[1] Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. 
(2018). Surrogate data for hypothesis testing of physical systems. 
Physics Reports, 748, 1-60.

[2] Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: 
uses and interpretations. Neuroimage, 52(3), 1059-1069.

"""

import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests


def get_uptri(x):
    """
    Gets the vectorized upper triangle of a matrix x.
    
    Parameters
    ----------
    x : numpy array.
        connectivity matrix.
    Returns
    -------
    vector : numpy array.
             upper triangle of x in vector form.
    """
    nnodes = x.shape[0]
    npairs = (nnodes**2 - nnodes) // 2
    vector = np.zeros(npairs)
    
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            vector[idx] = x[row, col]
            idx = idx + 1
    
    return(vector)


def matrix_recon(x):
    """
    Reconstructs a connectivity matrix from its upper triangle.
    
    Parameters
    ----------
    x : numpy array.
        upper triangle of the original matrix.
    Returns
    -------
    matrix : numpy array.
             original connectivity matrix.
    """
    npairs = len(x)
    nnodes = int((1 + np.sqrt(1 + 8 * npairs)) // 2)
    
    matrix = np.zeros((nnodes, nnodes))
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            matrix[row, col] = x[idx]
            idx = idx + 1
    matrix = matrix + matrix.T
   
    return(matrix)   
    


def probabilistic_thresholding(y, surr_number = 50, alpha = 0.05):
    """
    Threshold for neglecting spurious connectivity values in functional fata.
    It uses a phase randomization to destroy the pairwise correlations, while 
    preserving the spectral properties of the original signals. Type I error
    (multiple comparisons) corrected by the Benjamini-Hochberg procedure.
    
    Parameters
    ----------
    y : txN numpy array.
        time series.
        t -> time.
        N -> nodes.
    surr_number : integer.
                  number of surrogates.
    alpha : float (between 0 and 1).
            critical p_value for statistical inference. By default is 0.05.
    Returns
    -------
    FC_adjusted : NxN numpy array.
                  thresholded functional connectivity matrix.
    p_matrix : NxN numpy array.
               p_values for each connectivity pair.
    """

    y = y - np.mean(y, axis = 0)
    yf = np.fft.fft(y, axis = 0)

    nnodes = y.shape[1]
    npairs = (nnodes**2 - nnodes) // 2
    matrix_surr = np.zeros((npairs, surr_number))

    FC_real = np.corrcoef(y.T)
    FC_real = FC_real[np.triu_indices(n = nnodes, k = 1)]
    
    for i in range(surr_number):
        np.random.seed(i + 1)
        random_vector = np.random.uniform(0, 2 * np.pi, ((yf.shape[0] // 2), yf.shape[1]))
        random_vector = np.row_stack((random_vector, random_vector[::-1,:]))
        yfR = yf * np.exp(1j * random_vector)
        surrogate = np.fft.ifft(yfR, axis = 0)
        surrogate = surrogate.real
        FC_surr = np.corrcoef(surrogate.T)
        matrix_surr[:,i] = FC_surr[np.triu_indices(n = nnodes, k = 1)]

    p_vector = np.zeros(npairs)
    for i in range(npairs):
        mu, sigma = stats.norm.fit(matrix_surr[i,:])
        p_vector[i] = 1 - stats.norm.cdf(FC_real[i], mu, sigma)
    
    
    reject, p_adjust, alphacSidak, alphacBonf = multipletests(p_vector, alpha = alpha, method='fdr_bh')
    
    FC_adjusted = FC_real * ((p_adjust < alpha) * 1)
    FC_adjusted = matrix_recon(FC_adjusted)
    p_matrix = matrix_recon(p_adjust)
    
    return([FC_adjusted, p_matrix])
 
    
#Kuramoto order parameter
def simple_order_parameter(data):
    """
    Calculates the global phase synchronization over time as the Kuramoto order
    parameter.
    
    Parameters
    ----------
    data : phixN numpy array.
           phi -> phases.
           N -> nodes.
    Returns
    -------
    PhaseSynch : 1xphi numpy array.
                 Kuramoto order parameter.
    """
    phases = data
    PhaseSynch = np.abs(np.mean(np.exp(1j * phases), axis = 1))
    return(PhaseSynch)    
    
###################################    
###FUNCTIONS FOR GRAPH ANALYSIS###
###################################
###################################
class BCTParamError(RuntimeError):
    pass

def cuberoot(x):
    """
    Correctly handle the cube root for negative weights, instead of uselessly
    crashing as in python or returning the wrong root as in matlab
    """
    return(np.sign(x) * np.abs(x)**(1 / 3))    


def invert(W, copy = True):
    """
    Inverts elementwise the weights in an input connection matrix.
    In other words, change the from the matrix of internode strengths to the
    matrix of internode distances.
    If copy is not set, this function will *modify W in place.*
    Parameters
    ----------
    W : :obj:`numpy.ndarray`
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.
    Returns
    -------
    W : :obj:`numpy.ndarray`
        inverted connectivity matrix
    """
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1. / W[E]
    return(W)


def dummyvar(cis, return_sparse=False):
    """
    This is an efficient implementation of matlab's "dummyvar" command
    using sparse matrices.
    input: partitions, NxM array-like containing M partitions of N nodes
        into <=N distinct communities
    output: dummyvar, an NxR matrix containing R column variables (indicator
        variables) with N entries, where R is the total number of communities
        summed across each of the M partitions.
        i.e.
        r = sum((max(len(unique(partitions[i]))) for i in range(m)))
    """
    # num_rows is not affected by partition indexes
    n = np.size(cis, axis=0)
    m = np.size(cis, axis=1)
    r = np.sum((np.max(len(np.unique(cis[:, i])))) for i in range(m))
    nnz = np.prod(cis.shape)

    ix = np.argsort(cis, axis=0)
    # s_cis=np.sort(cis,axis=0)
    # FIXME use the sorted indices to sort by row efficiently
    s_cis = cis[ix][:, range(m), range(m)]

    mask = np.hstack((((True,),) * m, (s_cis[:-1, :] != s_cis[1:, :]).T))
    indptr, = np.where(mask.flat)
    indptr = np.append(indptr, nnz)

    import scipy.sparse as sp
    dv = sp.csc_matrix((np.repeat((1,), nnz), ix.T.flat, indptr), shape=(n, r))
    return dv.toarray()


def agreement(ci, buffsz=1000):
    """
    Takes as input a set of vertex partitions CI of
    dimensions [vertex x partition]. Each column in CI contains the
    assignments of each vertex to a class/community/module. This function
    aggregates the partitions in CI into a square [vertex x vertex]
    agreement matrix D, whose elements indicate the number of times any two
    vertices were assigned to the same class.
    In the case that the number of nodes and partitions in CI is large
    (greater than ~1000 nodes or greater than ~1000 partitions), the script
    can be made faster by computing D in pieces. The optional input BUFFSZ
    determines the size of each piece. Trial and error has found that
    BUFFSZ ~ 150 works well.
    Parameters
    ----------
    ci : NxM :obj:`numpy.ndarray`
        set of M (possibly degenerate) partitions of N nodes
    buffsz : int | None
        sets buffer size. If not specified, defaults to 1000
    Returns
    -------
    D : NxN :obj:`numpy.ndarray`
        agreement matrix
    """
    ci = np.array(ci)
    n_nodes, n_partitions = ci.shape

    if n_partitions <= buffsz:  # Case 1: Use all partitions at once
        ind = dummyvar(ci)
        D = np.dot(ind, ind.T)
    else:  # Case 2: Add together results from subsets of partitions
        a = np.arange(0, n_partitions, buffsz)
        b = np.arange(buffsz, n_partitions, buffsz)
        if len(a) != len(b):
            b = np.append(b, n_partitions)
        D = np.zeros((n_nodes, n_nodes))
        for i, j in zip(a, b):
            y = ci[:, i:j]
            ind = dummyvar(y)
            D += np.dot(ind, ind.T)

    np.fill_diagonal(D, 0)
    return D

  
    
def consensus_und(D, tau, reps = 1000):
    """
    This algorithm seeks a consensus partition of the
    agreement matrix D. The algorithm used here is almost identical to the
    one introduced in Lancichinetti & Fortunato (2012): The agreement
    matrix D is thresholded at a level TAU to remove an weak elements. The
    resulting matrix is then partitions REPS number of times using the
    Louvain algorithm (in principle, any clustering algorithm that can
    handle weighted matrixes is a suitable alternative to the Louvain
    algorithm and can be substituted in its place). This clustering
    produces a set of partitions from which a new agreement is built. If
    the partitions have not converged to a single representative partition,
    the above process repeats itself, starting with the newly built
    agreement matrix.
    NOTE: In this implementation, the elements of the agreement matrix must
    be converted into probabilities.
    NOTE: This implementation is slightly different from the original
    algorithm proposed by Lanchichinetti & Fortunato. In its original
    version, if the thresholding produces singleton communities, those
    nodes are reconnected to the network. Here, we leave any singleton
    communities disconnected.
    Parameters
    ----------
    D : NxN :obj:`numpy.ndarray`
        agreement matrix with entries between 0 and 1 denoting the probability
        of finding node i in the same cluster as node j
    tau : float
        threshold which controls the resolution of the reclustering
    reps : int
        number of times the clustering algorithm is reapplied. default value
        is 1000.
    Returns
    -------
    ciu : Nx1 :obj:`numpy.ndarray`
        consensus partition
    """
    def unique_partitions(cis):
        # relabels the partitions to recognize different numbers on same
        # topology
    
        n, r = np.shape(cis)  # ci represents one vector for each rep
        ci_tmp = np.zeros(n)
    
        for i in range(r):
            for j, u in enumerate(sorted(
                    np.unique(cis[:, i], return_index=True)[1])):
                ci_tmp[np.where(cis[:, i] == cis[u, i])] = j
            cis[:, i] = ci_tmp
            # so far no partitions have been deleted from ci
    
        # now squash any of the partitions that are completely identical
        # do not delete them from ci which needs to stay same size, so make
        # copy
        ciu = []
        cis = cis.copy()
        c = np.arange(r)
        # count=0
        while (c != 0).sum() > 0:
            ciu.append(cis[:, 0])
            dup = np.where(np.sum(np.abs(cis.T - cis[:, 0]), axis=1) == 0)
            cis = np.delete(cis, dup, axis=1)
            c = np.delete(c, dup)
        return np.transpose(ciu)
    
    n = len(D)
    flag = True
    while flag:
        flag = False
        dt = D * (D >= tau)
        np.fill_diagonal(dt, 0)
    
        if np.size(np.where(dt == 0)) == 0:
            ciu = np.arange(1, n + 1)
        else:
            cis = np.zeros((n, reps))
            for i in np.arange(reps):
                cis[:, i], _ = modularity_louvain_und_sign(dt)
            ciu = unique_partitions(cis)
            nu = np.size(ciu, axis=1)
            if nu > 1:
                flag = True
                D = agreement(cis) / reps
    
    return np.squeeze(ciu + 1)  



def transitivity_wu(W):
    """
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        weighted undirected connection matrix
    Returns
    -------
    T : int
        transitivity scalar
    """
    K = np.sum(np.logical_not(W == 0), axis=1)
    ws = cuberoot(W)
    cyc3 = np.diag(np.dot(ws, np.dot(ws, ws)))
    return np.sum(cyc3, axis=0) / np.sum(K * (K - 1), axis=0) 



def clustering_coef_wu(W):
    """
    The weighted clustering coefficient is the average "intensity" of
    triangles around a node.
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        weighted undirected connection matrix
    Returns
    -------
    C : Nx1 :obj:`numpy.ndarray`
        clustering coefficient vector
    """
    K = np.array(np.sum(np.logical_not(W == 0), axis=1), dtype=float)
    ws = cuberoot(W)
    cyc3 = np.diag(np.dot(ws, np.dot(ws, ws)))
    K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, set C=0
    C = cyc3 / (K * (K - 1))
    return C


def modularity_louvain_und_sign(W, gamma=1, qtype='sta', seed=None):
    """
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.
    The Louvain algorithm is a fast and accurate community detection
    algorithm (at the time of writing).
    Use this function as opposed to modularity_louvain_und() only if the
    network contains a mix of positive and negative weights.  If the network
    contains all positive weights, the output will be equivalent to that of
    modularity_louvain_und().
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        undirected weighted/binary connection matrix with positive and
        negative weights
    qtype : str
        modularity type. Can be 'sta' (default), 'pos', 'smp', 'gja', 'neg'.
        See Rubinov and Sporns (2011) for a description.
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.
    Returns
    -------
    ci : Nx1 :obj:`numpy.ndarray`
        refined community affiliation vector
    Q : float
        optimized modularity metric
    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    """
    np.random.seed(seed)

    n = len(W)  # number of nodes

    W0 = W * (W > 0)  # positive weights matrix
    W1 = -W * (W < 0)  # negative weights matrix
    s0 = np.sum(W0)  # weight of positive links
    s1 = np.sum(W1)  # weight of negative links

    if qtype == 'smp':
        d0 = 1 / s0
        d1 = 1 / s1  # dQ=dQ0/s0-sQ1/s1
    elif qtype == 'gja':
        d0 = 1 / (s0 + s1)
        d1 = d0  # dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype == 'sta':
        d0 = 1 / s0
        d1 = 1 / (s0 + s1)  # dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype == 'pos':
        d0 = 1 / s0
        d1 = 0  # dQ=dQ0/s0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1 / s1  # dQ=-dQ1/s1
    else:
        raise KeyError('modularity type unknown')

    if not s0:  # adjust for absent positive weights
        s0 = 1
        d0 = 0
    if not s1:  # adjust for absent negative weights
        s1 = 1
        d1 = 0

    h = 1  # hierarchy index
    nh = n  # number of nodes in hierarchy
    ci = [None, np.arange(n) + 1]  # hierarchical module assignments
    q = [-1, 0]  # hierarchical modularity values
    while q[h] - q[h - 1] > 1e-10:
        if h > 300:
            raise BCTParamError('Modularity Infinite Loop Style A.  Please '
                                'contact the developer with this error.')
        kn0 = np.sum(W0, axis=0)  # positive node degree
        kn1 = np.sum(W1, axis=0)  # negative node degree
        km0 = kn0.copy()  # positive module degree
        km1 = kn1.copy()  # negative module degree
        knm0 = W0.copy()  # positive node-to-module degree
        knm1 = W1.copy()  # negative node-to-module degree

        m = np.arange(nh) + 1  # initial module assignments
        flag = True  # flag for within hierarchy search
        it = 0
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Infinite Loop was detected and stopped. '
                                    'This was probably caused by passing in a '
                                    'directed matrix.')
            flag = False
            # loop over nodes in random order
            for u in np.random.permutation(nh):
                ma = m[u] - 1
                # positive dQ
                dQ0 = ((knm0[u, :] + W0[u, u] - knm0[u, ma]) -
                       gamma * kn0[u] * (km0 + kn0[u] - km0[ma]) / s0)
                # negative dQ
                dQ1 = ((knm1[u, :] + W1[u, u] - knm1[u, ma]) -
                       gamma * kn1[u] * (km1 + kn1[u] - km1[ma]) / s1)

                dQ = d0 * dQ0 - d1 * dQ1  # rescaled changes in modularity
                dQ[ma] = 0  # no changes for same module

                max_dQ = np.max(dQ)  # maximal increase in modularity
                if max_dQ > 1e-10:  # if maximal increase is positive
                    flag = True
                    mb = np.argmax(dQ)

                    # change positive node-to-module degrees
                    knm0[:, mb] += W0[:, u]
                    knm0[:, ma] -= W0[:, u]
                    # change negative node-to-module degrees
                    knm1[:, mb] += W1[:, u]
                    knm1[:, ma] -= W1[:, u]
                    km0[mb] += kn0[u]  # change positive module degrees
                    km0[ma] -= kn0[u]
                    km1[mb] += kn1[u]  # change negative module degrees
                    km1[ma] -= kn1[u]

                    m[u] = mb + 1  # reassign module

        h += 1
        ci.append(np.zeros((n,)))
        _, m = np.unique(m, return_inverse=True)
        m += 1

        for u in range(nh):  # loop through initial module assignments
            ci[h][np.where(ci[h - 1] == u + 1)] = m[u]  # assign new modules

        nh = np.max(m)  # number of new nodes
        wn0 = np.zeros((nh, nh))  # new positive weights matrix
        wn1 = np.zeros((nh, nh))

        for u in range(nh):
            for v in range(u, nh):
                wn0[u, v] = np.sum(W0[np.ix_(m == u + 1, m == v + 1)])
                wn1[u, v] = np.sum(W1[np.ix_(m == u + 1, m == v + 1)])
                wn0[v, u] = wn0[u, v]
                wn1[v, u] = wn1[u, v]

        W0 = wn0
        W1 = wn1

        q.append(0)
        # compute modularity
        q0 = np.trace(W0) - np.sum(np.dot(W0, W0)) / s0
        q1 = np.trace(W1) - np.sum(np.dot(W1, W1)) / s1
        q[h] = d0 * q0 - d1 * q1

    _, ci_ret = np.unique(ci[-1], return_inverse=True)
    ci_ret += 1

    return ci_ret, q[-1]


def efficiency_wei(Gw, local = False):
    """
    The global efficiency is the average of inverse shortest path length,
    and is inversely related to the characteristic path length.
    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.
    Parameters
    ----------
    Gw : NxN :obj:`numpy.ndarray`
        undirected weighted connection matrix
        (all weights in W must be between 0 and 1)
    local : bool
        If True, computes local efficiency instead of global efficiency.
        Default value = False.
    Returns
    -------
    Eglob : float
        global efficiency, only if local=False
    Eloc : Nx1 :obj:`numpy.ndarray`
        local efficiency, only if local=True
    Notes
    -----
       The  efficiency is computed using an auxiliary connection-length
    matrix L, defined as L_ij = 1/W_ij for all nonzero L_ij; This has an
    intuitive interpretation, as higher connection weights intuitively
    correspond to shorter lengths.
       The weighted local efficiency broadly parallels the weighted
    clustering coefficient of Onnela et al. (2005) and distinguishes the
    influence of different paths based on connection weights of the
    corresponding neighbors to the node in question. In other words, a path
    between two neighbors with strong connections to the node in question
    contributes more to the local efficiency than a path between two weakly
    connected neighbors. Note that this weighted variant of the local
    efficiency is hence not a strict generalization of the binary variant.
    Algorithm:  Dijkstra's algorithm
    """
    def distance_inv_wei(G):
        n = len(G)
        D = np.zeros((n, n))  # distance matrix
        D[np.logical_not(np.eye(n))] = np.inf

        for u in range(n):
            # distance permanence (true is temporary)
            S = np.ones((n,), dtype=bool)
            G1 = G.copy()
            V = [u]
            while True:
                S[V] = 0  # distance u->V is now permanent
                G1[:, V] = 0  # no in-edges as already shortest
                for v in V:
                    W, = np.where(G1[v, :])  # neighbors of smallest nodes
                    td = np.array(
                        [D[u, W].flatten(), (D[u, v] + G1[v, W]).flatten()])
                    D[u, W] = np.min(td, axis=0)

                if D[u, S].size == 0:  # all nodes reached
                    break
                minD = np.min(D[u, S])
                if np.isinf(minD):  # some nodes cannot be reached
                    break
                V, = np.where(D[u, :] == minD)

        np.fill_diagonal(D, 1)
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    n = len(Gw)
    Gl = invert(Gw, copy=True)  # connection length matrix
    A = np.array((Gw != 0), dtype=int)
    if local:
        E = np.zeros((n,))  # local efficiency
        for u in range(n):
            # find pairs of neighbors
            V, = np.where(np.logical_or(Gw[u, :], Gw[:, u].T))
            # symmetrized vector of weights
            sw = cuberoot(Gw[u, V]) + cuberoot(Gw[V, u].T)
            # inverse distance matrix
            e = distance_inv_wei(Gl[np.ix_(V, V)])
            # symmetrized inverse distance matrix
            se = cuberoot(e) + cuberoot(e.T)

            numer = np.sum(np.outer(sw.T, sw) * se) / 2
            if numer != 0:
                # symmetrized adjacency vector
                sa = A[u, V] + A[V, u].T
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                # print numer,denom
                E[u] = numer / denom  # local efficiency

    else:
        e = distance_inv_wei(Gl)
        E = np.sum(e) / (n * n - n)
    return(E)  
    
    
    
def charpath(D, include_diagonal=False, include_infinite=True):
    """
    The characteristic path length is the average shortest path length in
    the network. The global efficiency is the average inverse shortest path
    length in the network.
    Parameters
    ----------
    D : NxN :obj:`numpy.ndarray`
        distance matrix
    include_diagonal : bool
        If True, include the weights on the diagonal. Default value is False.
    include_infinite : bool
        If True, include infinite distances in calculation
    Returns
    -------
    lambda : float
        characteristic path length
    efficiency : float
        global efficiency
    ecc : Nx1 :obj:`numpy.ndarray`
        eccentricity at each vertex
    radius : float
        radius of graph
    diameter : float
        diameter of graph
    Notes
    -----
    The input distance matrix may be obtained with any of the distance
    functions, e.g. distance_bin, distance_wei.
    Characteristic path length is calculated as the global mean of
    the distance matrix D, excludings any 'Infs' but including distances on
    the main diagonal.
    """
    D = D.copy()

    if not include_diagonal:
        np.fill_diagonal(D, np.nan)

    if not include_infinite:
        D[np.isinf(D)] = np.nan

    Dv = D[np.logical_not(np.isnan(D))].ravel()

    # mean of finite entries of D[G]
    lambda_ = np.mean(Dv)

    # efficiency: mean of inverse entries of D[G]
    efficiency = np.mean(1 / Dv)

    # eccentricity for each vertex (ignore inf)
    ecc = np.array(np.ma.masked_where(np.isnan(D), D).max(axis=1))

    # radius of graph
    radius = np.min(ecc)  # but what about zeros?

    # diameter of graph
    diameter = np.max(ecc)

    return lambda_, efficiency, ecc, radius, diameter      



def community_louvain(W, gamma= 1, ci=None, B='modularity', seed=None):
    """
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes which maximizes the number of within-group
    edges and minimizes the number of between-group edges.
    This function is a fast an accurate multi-iterative generalization of the
    louvain community detection algorithm. This function subsumes and improves
    upon modularity_[louvain,finetune]_[und,dir]() and additionally allows to
    optimize other objective functions (includes built-in Potts Model i
    Hamiltonian, allows for custom objective-function matrices).
    Parameters
    ----------
    W : NxN np.array
        directed/undirected weighted/binary adjacency matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
        ignored if an objective function matrix is specified.
    ci : Nx1 np.arraylike
        initial community affiliation vector. default value=None
    B : str | NxN np.arraylike
        string describing objective function type, or provides a custom
        NxN objective-function matrix. builtin values
            'modularity' uses Q-metric as objective function
            'potts' uses Potts model Hamiltonian.
            'negative_sym' symmetric treatment of negative weights
            'negative_asym' asymmetric treatment of negative weights
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.
    Returns
    -------
    ci : Nx1 np.array
        final community structure
    q : float
        optimized q-statistic (modularity only)
    """
    np.random.seed(seed)

    n = len(W)
    s = np.sum(W)

    if np.min(W) < -1e-10:
        raise BCTParamError('adjmat must not contain negative weights')

    if ci is None:
        ci = np.arange(n) + 1
    else:
        if len(ci) != n:
            raise BCTParamError('initial ci vector size must equal N')
        _, ci = np.unique(ci, return_inverse=True)
        ci += 1
    Mb = ci.copy()

    if B in ('negative_sym', 'negative_asym'):
        W0 = W * (W > 0)
        s0 = np.sum(W0)
        B0 = W0 - gamma * np.outer(np.sum(W0, axis=1), np.sum(W, axis=0)) / s0

        W1 = W * (W < 0)
        s1 = np.sum(W1)
        if s1:
            B1 = (W1 - gamma * np.outer(np.sum(W1, axis=1), np.sum(W1, axis=0))
                  / s1)
        else:
            B1 = 0

    elif np.min(W) < -1e-10:
        raise BCTParamError('Input connection matrix contains negative '
                            'weights but objective function dealing with '
                            'negative weights was not selected')

    if B == 'potts' and np.any(np.logical_not(np.logical_or(W == 0, W == 1))):
        raise BCTParamError('Potts hamiltonian requires binary input matrix')

    if B == 'modularity':
        B = W - gamma * np.outer(np.sum(W, axis=1), np.sum(W, axis=0)) / s
    elif B == 'potts':
        B = W - gamma * np.logical_not(W)
    elif B == 'negative_sym':
        B = B0 / (s0 + s1) - B1 / (s0 + s1)
    elif B == 'negative_asym':
        B = B0 / s0 - B1 / (s0 + s1)
    else:
        try:
            B = np.array(B)
        except BCTParamError:
            print('unknown objective function type')

        if B.shape != W.shape:
            raise BCTParamError('objective function matrix does not match '
                                'size of adjacency matrix')
        if not np.allclose(B, B.T):
            print('Warning: objective function matrix not symmetric, '
                  'symmetrizing')
            B = (B + B.T) / 2

    Hnm = np.zeros((n, n))
    for m in range(1, n + 1):
        Hnm[:, m - 1] = np.sum(B[:, ci == m], axis=1)  # node to module degree
    H = np.sum(Hnm, axis=1)  # node degree
    Hm = np.sum(Hnm, axis=0)  # module degree

    q0 = -np.inf
    # compute modularity
    q = np.sum(B[np.tile(ci, (n, 1)) == np.tile(ci, (n, 1)).T]) / s

    first_iteration = True

    while q - q0 > 1e-10:
        it = 0
        flag = True
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Modularity infinite loop style G. '
                                    'Please contact the developer.')
            flag = False
            for u in np.random.permutation(n):
                ma = Mb[u] - 1
                dQ = Hnm[u, :] - Hnm[u, ma] + B[u, u]  # algorithm condition
                dQ[ma] = 0

                max_dq = np.max(dQ)
                if max_dq > 1e-10:
                    flag = True
                    mb = np.argmax(dQ)

                    Hnm[:, mb] += B[:, u]
                    Hnm[:, ma] -= B[:, u]  # change node-to-module strengths

                    Hm[mb] += H[u]
                    Hm[ma] -= H[u]  # change module strengths

                    Mb[u] = mb + 1

        _, Mb = np.unique(Mb, return_inverse=True)
        Mb += 1

        M0 = ci.copy()
        if first_iteration:
            ci = Mb.copy()
            first_iteration = False
        else:
            for u in range(1, n + 1):
                ci[M0 == u] = Mb[u - 1]  # assign new modules

        n = np.max(Mb)
        b1 = np.zeros((n, n))
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                # pool weights of nodes in same module
                bm = np.sum(B[np.ix_(Mb == i, Mb == j)])
                b1[i - 1, j - 1] = bm
                b1[j - 1, i - 1] = bm
        B = b1.copy()

        Mb = np.arange(1, n + 1)
        Hnm = B.copy()
        H = np.sum(B, axis=0)
        Hm = H.copy()

        q0 = q
        q = np.trace(B) / s  # compute modularity

    return ci, q


def get_agreement_matrix(W, reps = 100, tau = 0.5, gamma = 1, ci = None, B = 'modularity', seeds = None):
    nnodes = W.shape[0]
    n_p = reps #number of partitions
    D_matrix = np.zeros((nnodes,nnodes)) #Agreement matrix
    
    if seeds == None:
        seed_vector = np.arange(0, n_p, 1)
    elif len(seed_vector) == n_p:
        seed_vector = seeds
    else:
        print('seeds must be an array of length equal to reps, or None')
    
    for rand in range(0,n_p):
        partition = community_louvain(W, gamma, ci, B, seed_vector[rand])[0]
        for row in range(0,nnodes):
            for col in range(0,nnodes):
                D_matrix[row,col] += (partition[row] == partition[col]) * 1
    D_matrix /= n_p
    D_matrix[D_matrix < tau] = 0        
    np.fill_diagonal(D_matrix, 0)
    
    return(D_matrix)


def distance_wei(G):
    """
    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path
    from node u to node v. The average shortest path length is the
    characteristic path length of the network.
    Parameters
    ----------
    G : NxN :obj:`numpy.ndarray`
        Directed/undirected connection-length matrix.
        NB L is not the adjacency matrix. See below.
    Returns
    -------
    D : NxN :obj:`numpy.ndarray`
        distance (shortest weighted path) matrix
    B : NxN :obj:`numpy.ndarray`
        matrix of number of edges in shortest weighted path
    Notes
    -----
       The input matrix must be a connection-length matrix, typically
    obtained via a mapping from weight to length. For instance, in a
    weighted correlation network higher correlations are more naturally
    interpreted as shorter distances and the input matrix should
    consequently be some inverse of the connectivity matrix.
       The number of edges in shortest weighted paths may in general
    exceed the number of edges in shortest binary paths (i.e. shortest
    paths computed on the binarized connectivity matrix), because shortest
    weighted paths have the minimal weighted distance, but not necessarily
    the minimal number of edges.
       Lengths between disconnected nodes are set to Inf.
       Lengths on the main diagonal are set to 0.
    Algorithm: Dijkstra's algorithm.
    """
    n = len(G)
    D = np.zeros((n, n))  # distance matrix
    D[np.logical_not(np.eye(n))] = np.inf
    B = np.zeros((n, n))  # number of edges matrix

    for u in range(n):
        # distance permanence (true is temporary)
        S = np.ones((n,), dtype=bool)
        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                W, = np.where(G1[v, :])  # neighbors of shortest nodes

                td = np.array(
                    [D[u, W].flatten(), (D[u, v] + G1[v, W]).flatten()])
                d = np.min(td, axis=0)
                wi = np.argmin(td, axis=0)

                D[u, W] = d  # smallest of old/new path lengths
                ind = W[np.where(wi == 1)]  # indices of lengthened paths
                # increment nr_edges for lengthened paths
                B[u, ind] = B[u, v] + 1

            if D[u, S].size == 0:  # all nodes reached
                break
            minD = np.min(D[u, S])
            if np.isinf(minD):  # some nodes cannot be reached
                break

            V, = np.where(D[u, :] == minD)

    return D, B



def participation_coef(W, ci, degree = 'undirected'):
    """
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        binary/weighted directed/undirected connection matrix
    ci : Nx1 :obj:`numpy.ndarray`
        community affiliation vector
    degree : {'undirected', 'in', 'out'}, optional
        Flag to describe nature of graph. 'undirected': For undirected graphs,
        'in': Uses the in-degree, 'out': Uses the out-degree
    Returns
    -------
    P : Nx1 :obj:`numpy.ndarray`
        participation coefficient
    """
    if degree == 'in':
        W = W.T

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return(P)
    
    
def modularity_und(A, gamma=1, kci=None):
    """
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.
    Parameters
    ----------
    A : NxN :obj:`numpy.ndarray`
        undirected weighted/binary connection matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    kci : Nx1 :obj:`numpy.ndarray` | None
        starting community structure. If specified, calculates the Q-metric
        on the community structure giving, without doing any optimzation.
        Otherwise, if not specified, uses a spectral modularity maximization
        algorithm.
    Returns
    -------
    ci : Nx1 :obj:`numpy.ndarray`
        optimized community structure
    Q : float
        maximized modularity metric
    Notes
    -----
    This algorithm is deterministic. The matlab function bearing this
    name incorrectly disclaims that the outcome depends on heuristics
    involving a random seed. The louvain method does depend on a random seed,
    but this function uses a deterministic modularity maximization algorithm.
    """
    from scipy import linalg
    n = len(A)  # number of vertices
    k = np.sum(A, axis=0)  # degree
    m = np.sum(k)  # number of edges (each undirected edge
    # is counted twice)
    B = A - gamma * np.outer(k, k) / m  # initial modularity matrix

    init_mod = np.arange(n)  # initial one big module
    modules = []  # output modules list

    def recur(module):
        n = len(module)
        modmat = B[module][:, module]
        modmat -= np.diag(np.sum(modmat, axis=0))

        vals, vecs = linalg.eigh(modmat)  # biggest eigendecomposition
        rlvals = np.real(vals)
        max_eigvec = np.squeeze(vecs[:, np.where(rlvals == np.max(rlvals))])
        if max_eigvec.ndim > 1:  # if multiple max eigenvalues, pick one
            max_eigvec = max_eigvec[:, 0]
        # initial module assignments
        mod_asgn = np.squeeze((max_eigvec >= 0) * 2 - 1)
        q = np.dot(mod_asgn, np.dot(modmat, mod_asgn))  # modularity change

        if q > 0:  # change in modularity was positive
            qmax = q
            np.fill_diagonal(modmat, 0)
            it = np.ma.masked_array(np.ones((n,)), False)
            mod_asgn_iter = mod_asgn.copy()
            while np.any(it):  # do some iterative fine tuning
                # this line is linear algebra voodoo
                q_iter = qmax - 4 * mod_asgn_iter * \
                    (np.dot(modmat, mod_asgn_iter))
                qmax = np.max(q_iter * it)
                imax = np.argmax(q_iter * it)
                # imax, = np.where(q_iter == qmax)
                # if len(imax) > 1:
                #     imax = imax[0]
                # does switching increase modularity?
                mod_asgn_iter[imax] *= -1
                it[imax] = np.ma.masked
                if qmax > q:
                    q = qmax
                    mod_asgn = mod_asgn_iter
            if np.abs(np.sum(mod_asgn)) == n:  # iteration yielded null module
                modules.append(np.array(module).tolist())
                return
            else:
                mod1 = module[np.where(mod_asgn == 1)]
                mod2 = module[np.where(mod_asgn == -1)]

                recur(mod1)
                recur(mod2)
        else:  # change in modularity was negative or 0
            modules.append(np.array(module).tolist())

    # adjustment to one-based indexing occurs in ls2ci
    if kci is None:
        recur(init_mod)
        ci = ls2ci(modules)
    else:
        ci = kci
    s = np.tile(ci, (n, 1))
    q = np.sum(np.logical_not(s - s.T) * B / m)
    return ci, q


def ls2ci(ls, zeroindexed=False):
    """
    Convert from a 2D python list of modules to a community index vector.
    The list is a pure python list, not requiring numpy.
    Parameters
    ----------
    ls : listof(list)
        pure python list with lowest value zero-indexed
        (regardless of value of zeroindexed parameter)
    zeroindexed : bool
        If True, ci uses zero-indexing (lowest value is 0). Defaults to False.
    Returns
    -------
    ci : Nx1 :obj:`numpy.ndarray`
        community index vector
    """
    if ls is None or np.size(ls) == 0:
        return ()  # list is empty
    nr_indices = sum(map(len, ls))
    ci = np.zeros((nr_indices,), dtype=int)
    z = int(not zeroindexed)
    for i, x in enumerate(ls):
        for j, y in enumerate(ls[i]):
            ci[ls[i][j]] = i + z
    return ci


def rich_club_wu(CIJ, klevel=None):
    """
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        weighted undirected connection matrix
    klevel : int | None
        sets the maximum level at which the rich club coefficient will be
        calculated. If None (default), the maximum level is set to the
        maximum degree of the adjacency matrix
    Returns
    -------
    Rw : Kx1 :obj:`numpy.ndarray`
        vector of rich-club coefficients for levels 1 to klevel
    """
    # nr_nodes = len(CIJ)
    deg = np.sum((CIJ != 0), axis=0)

    if klevel is None:
        klevel = np.max(deg)
    Rw = np.zeros((klevel,))

    # sort the weights of the network, with the strongest connection first
    wrank = np.sort(CIJ.flat)[::-1]
    
    SmallNodes_all = []
    for k in range(klevel):
        SmallNodes, = np.where(deg < k + 1)
        SmallNodes_all.append(SmallNodes)
#        if np.size(SmallNodes) == 0:
#            Rw[k] = np.nan
#            continue

        # remove small nodes with node degree < k
        cutCIJ = np.delete(
            np.delete(CIJ, SmallNodes, axis=0), SmallNodes, axis=1)
        # total weight of connections in subset E>r
        Wr = np.sum(cutCIJ)
        # total number of connections in subset E>r
        Er = np.size(np.where(cutCIJ.flat != 0), axis=1)
        # E>r number of connections with max weight in network
        wrank_r = wrank[:Er]
        # weighted rich-club coefficient
        Rw[k] = Wr / np.sum(wrank_r)
    return ([Rw,SmallNodes_all])


def score_wu(CIJ, s):
    """
    The s-core is the largest subnetwork comprising nodes of strength at
    least s. This function computes the s-core for a given weighted
    undirected connection matrix. Computation is analogous to the more
    widely used k-core, but is based on node strengths instead of node
    degrees.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        weighted undirected connection matrix
    s : float
        level of s-core. Note that can take on any fractional value.
    Returns
    -------
    CIJscore : NxN :obj:`numpy.ndarray`
        connection matrix of the s-core. This matrix contains only nodes with
        a strength of at least s.
    sn : int
        size of s-core
    """
    CIJscore = CIJ.copy()
    while True:
        str = strengths_und(CIJscore)  # get strengths of matrix

        # find nodes with strength <s
        ff, = np.where(np.logical_and(str < s, str > 0))

        if ff.size == 0:
            break  # if none found -> stop

        # else peel away found nodes
        CIJscore[ff, :] = 0
        CIJscore[:, ff] = 0

    sn = np.sum(str > 0)
    return CIJscore, sn


def strengths_und(CIJ):
    """
    Node strength is the sum of weights of links connected to the node.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        undirected weighted connection matrix
    Returns
    -------
    str : Nx1 :obj:`numpy.ndarray`
        node strengths
    """
    return np.sum(CIJ, axis=0)


    