import matplotlib.pyplot as plt
import numpy as np

def ensamble_avarage(order_parameter: np.array, lambd: np.array, beta_range: np.array):
    degree_ens = np.zeros_like(beta_range)
    Z_arr = np.zeros_like(beta_range)
    lambd = lambd # - lambd.min()
    for i, b in enumerate(beta_range):
        lrho = np.exp(-b * lambd)
        Z = lrho.sum()
        Z_arr[i] = Z
        if Z == 0:
            degree_ens[i] = 0
        else:
            degree_ens[i] = (order_parameter * lrho).sum() / Z
    return degree_ens, Z_arr



def ensamble_avarage(order_parameter: np.array, lambd: np.array, beta_range: np.array):
    degree_ens = np.zeros_like(beta_range)
    Z_arr = np.zeros_like(beta_range)
    lambd = lambd - lambd.min()
    for i, b in enumerate(beta_range):
        lrho = np.exp(-b * lambd)
        Z = lrho.sum()
        Z_arr[i] = Z
        if Z == 0:
            degree_ens[i] = 0
        else:
            degree_ens[i] = (order_parameter * lrho).sum() / Z
    return degree_ens, Z_arr

# TODO
def ensamble_average_batch(order_parameter: np.array, lambd: np.array, tau_range: np.array):
    lambd = lambd  # Normalizzazione di lambda per ogni batch
    lrho = np.exp(-np.multiply.outer(tau_range, lambd))  # Calcolo di lrho per ogni valore di beta e per ogni batch
    print(lrho.shape)
    Z_arr = lrho.sum(axis=2)  # Calcolo di Z per ogni batch
    degree_ens = np.sum(np.multiply.outer(order_parameter, lrho),
                        axis=2) / Z_arr  # Calcolo del grado medio per ogni batch
    return degree_ens, Z_arr
