

from scipy.sparse.linalg import eigs, eigsh
from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import expm, logm, eigvalsh
from scipy.stats import entropy
from scipy.sparse import coo_matrix
from typing import Optional, Tuple
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx


def entropy_var_ensemble(lambd: np.array, beta_range: np.array):
    entropy = np.zeros_like(beta_range)
    entropy_2 = np.zeros_like(beta_range)
    Z_arr = np.zeros_like(beta_range)
    lambd = lambd #- lambd.min()
    for i, b in enumerate(beta_range):
        lrho = np.exp(-b * lambd)
        Z = lrho.sum()
        Z_arr[i] = Z
        lrho = np.delete(lrho, np.where(lrho < 10 ** -12))
        Z = np.sum(lrho)
        p = lrho / Z
        entropy[i] = np.sum(-p * np.log(p))
        entropy_2[i] = np.sum(p * (np.log(p)**2))
    var_ent = entropy_2 - entropy ** 2
    # from matplotlib import pyplot as plt
    # plt.semilogx(beta_range, entropy)
    # plt.semilogx(beta_range, var_ent)
    # plt.show()
    return entropy, var_ent, Z_arr


def specific_heat(spec_ent: np.array, tau_range: np.array, batch_dim: int=0) -> np.array:
    """
    Calculate the specific heat given the specific entropy and temperature range.

    Parameters:
    -----------
    spec_ent : np.ndarray
        Array containing specific entropy values. It should have shape [batch_size, num_points] if batch_dim=0,
        or [num_points, batch_size] if batch_dim=1.

    tau_range : np.ndarray
        Array containing temperature values corresponding to each specific entropy point.

    batch_dim : int, optional
        The axis representing the batch dimension in spec_ent. Default is 0.

    Returns:
    --------
    np.ndarray
        Array containing specific heat values.
    """

    if batch_dim == 0 and len(spec_ent.shape) == 2:
        C = - np.gradient(spec_ent, np.log(tau_range), axis=1)
    elif batch_dim == 1 or len(spec_ent.shape) == 1:
        C = - np.gradient(spec_ent, np.log(tau_range), axis=0)
    else:
        raise ValueError("Invalid batch_dim value. Must be 0 or 1.")

    return C


def von_neumann_entropy_numpy(tau_range: np.array, L: np.array = None, lambd: np.array = None):
    """
    Adapted from https://networkqit.github.io/

    This function computes Von Neumann spectral entropy over a range of tau values
    for a (batched) network with Laplacian L
    :math:`S(\\rho) = -\\mathrm{Tr}[\\rho \\log \\rho]`

    Parameters
    ----------

    tau_range: (iterable) list or numpy.array
        The range of tau

    L: np.array, optional
        The (batched) n x n graph laplacian. If batched, the input dimension is [batch_size,n,n]

    lambd: np.array, optional
        The eigenvalues of L. If provided, entropy computation will be skipped.
        Dimensions: (batched) x n

    Returns
    -------
    np.array
        The unnormalized Von Neumann entropy over the tau values and over all batches.
        Final dimension is [b,batch_size]. If 2D input, final dimension is [b]
        where b is the number of elements in the array beta_range

    Raises
    ------
    ValueError
        If neither L nor lambd is provided.
    """

    if lambd is None:
        if L is None:
            raise ValueError("Either 'L' or 'lambd' must be provided.")
        lambd, Q = np.linalg.eigh(L)  # eigenvalues and eigenvectors of (batched) Laplacian

    ndim = len(lambd.shape)
    if ndim == 2:
        # batch_size = lambd.shape[0]
        lrho = np.exp(-np.multiply.outer(tau_range, lambd))
        Z = np.sum(lrho, axis=2)
        entropy = np.log(Z) + tau_range[:, None] * np.sum(lambd * lrho, axis=2) / Z
    elif ndim == 1:
        entropy = np.zeros_like(tau_range)
        for i, b in enumerate(tau_range):
            lrho = np.exp(-b * lambd)
            Z = lrho.sum()
            entropy[i] = np.log(np.abs(Z)) + b * (lambd * lrho).sum() / Z
    else:
        raise RuntimeError('Must provide a 2D or 3D array (as batched 2D arrays)')
    entropy[np.isnan(entropy)] = 0

    return entropy, Z, lambd

