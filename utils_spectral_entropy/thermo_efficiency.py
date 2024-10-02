import numpy as np
import pandas as pd

def thermo_efficiency_by_key(tau_range: np.array, spectrum: np.array) -> (np.array, np.array,
                                                                          np.array, np.array,
                                                                          np.array, np.array
                                                                          ):
    """
        From Gavasieh, De Domenico, Nat.Phys. 2024

        Computes thermodynamic efficiency based on a range of tau values and DataFrame columns specified by keys.

        Parameters:
        -----------
        tau_range : np.array
            Array of tau values.

        df : pd.DataFrame
            DataFrame containing the data.

        keys : str
            Column keys in the DataFrame for which thermodynamic efficiency will be computed.

        Returns:
        --------
        networks_eta_avg : np.array
            Array containing thermodynamic efficiency corresponding to each tau value averaged over the independent \
            runs in the DataFrame.

        networks_eta_std : np.array
            Array containing the std of thermodynamic efficiency corresponding to each tau value over the independent \
            runs in the DataFrame.
        """

    networks_eta_avg = []
    networks_eta_std = []
    dS_avg = []
    dS_std = []
    dF_avg = []
    dF_std = []

    # for d in df['R'].unique():
    for r_ind in range(spectrum.shape[1]):
        temp_eta = []
        temp_dS = []
        temp_dF = []
        # Loop over indipendent run to compute the avg and std of eta
        for sam in spectrum[:, r_ind]:  # iterate over rows with iterrows
            # print(sam['spectrum'].shape)
            dS_, dF_, eta_ = thermo_trajectory(sam, tau_range)
            # eta_ = thermo_trajectory(sam, tau_range)[-1]
            temp_eta.append(eta_)
            temp_dS.append(dS_)
            temp_dF.append(dF_)

        networks_eta_avg.append(np.mean(temp_eta, axis=0))
        networks_eta_std.append(np.std(temp_eta, axis=0))

        dS_avg.append(np.mean(temp_dS, axis=0))
        dS_std.append(np.std(temp_dS, axis=0))

        dF_avg.append(np.mean(temp_dF, axis=0))
        dF_std.append(np.std(temp_dF, axis=0))

    networks_eta_avg = np.array(networks_eta_avg).T
    networks_eta_std = np.array(networks_eta_std).T

    dS_avg = np.array(dS_avg).T
    dS_std = np.array(dS_std).T

    dF_avg = np.array(dF_avg).T
    dF_std = np.array(dF_std).T

    return (networks_eta_avg, networks_eta_std,
            dS_avg, dS_std,
            dF_avg, dF_std)

def thermo_spectrum(spectrum, tau):
    """
    inputs : the Laplacian spectrum, the propagation scale tau
    outputs : the change in entropy (dS), free energy (dF) and the eta of network formation
    """

    N = len(spectrum)
    # Nc = len(np.where(spectrum<10**-10)[0])
    p = np.exp(-tau * spectrum)
    spectrum = np.delete(spectrum, np.where(p < 10 ** -12))
    p = np.delete(p, np.where(p < 10 ** -12))
    Z = np.sum(p)
    p = p / Z
    dF = (np.log(N) - np.log(Z))
    dS = np.sum(-p * np.log(p)) - np.log(N)
    eta = (dF + dS) / dF
    return dS, dF, eta


def thermo_trajectory(Ls, tau_list):
    """
    inputs : the Laplacian spectrum, a list of propagation scales tau (also indicated as tau)
    outputs : lists indicating the change in entropy (dS), free energy (dF) and the eta of network formation,
    at each propagation scale
    """
    n = len(tau_list)
    dS_ = np.zeros(n)
    dF_ = np.zeros(n)
    eta_ = np.zeros(n)
    for i in range(n):
        beta = tau_list[i]
        dS_[i], dF_[i], eta_[i] = thermo_spectrum(Ls, beta)
    return dS_, dF_, eta_
