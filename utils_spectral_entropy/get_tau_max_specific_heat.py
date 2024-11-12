import numpy as np
from matplotlib import pyplot as plt

from utils_spectral_entropy.utils import find_peaks_indices
from utils_spectral_entropy.spectral_entropy_numpy import von_neumann_entropy_numpy, specific_heat
from utils_spectral_entropy.make_plots import plot_von_neumann_ent


def get_tau_max_specific_heat(spectrum, tau_range, r_list):
    tau_error_peak_list = []

    tau_firstpeak = []
    tau_lastpeak = []
    tau_maxpeak = []

    critical_entropy_firstpeak = []
    critical_entropy_lastpeak = []
    critical_entropy_maxpeak = []

    spec_heat_firstpeak = []
    spec_heat_lastpeak = []
    spec_heat_maxpeak = []

    entropy_save = []
    C_save = []

    mask_max_list = []
    # critical_entropy = []
    spec_heat_c = []

    mask_list = []
    for key in r_list:
    # entropy, Z, _ = von_neumann_entropy_numpy_isotherm_isobar(tau_range=tau_range,
        #                                                           lambd=spectrum[:, r_ == key].squeeze(),
        #                                                           volume=volume[:, r_ == key].squeeze())
        entropy, Z, _ = von_neumann_entropy_numpy(tau_range=tau_range, lambd=spectrum[:, r_list == key].squeeze())
        entropy = entropy / np.log(spectrum.shape[-1])
        entropy = np.clip(a=entropy, a_min=0, a_max=1e15)
        entropy_save.append(entropy)

        C = np.log(spectrum.shape[-1])*np.clip(a_min=0, a_max=1e15, a=specific_heat(spec_ent=entropy, tau_range=tau_range, batch_dim=1))
        C_save.append(C)

        eps = 1e-18
        atol = 5e-10
        mask = find_peaks_indices(x_array=tau_range,
                                  y_array=C.mean(axis=1) + C.std(axis=1),
                                  #-np.gradient(entropy, np.log(tau_range), axis=0).mean(axis=1),
                                  eps=eps, atol=atol)
        mask_list.append(mask)
        tau_firstpeak.append(tau_range[mask].min())
        tau_lastpeak.append(tau_range[mask].max())
        tau_maxpeak.append(tau_range[np.argmax(C.mean(axis=1)[mask])])

        critical_entropy_firstpeak.append(entropy.mean(axis=1)[np.min(mask)])
        critical_entropy_lastpeak.append(entropy.mean(axis=1)[np.max(mask)])
        critical_entropy_maxpeak.append(entropy.mean(axis=1)[np.argmax(C.mean(axis=1)[mask])])

        spec_heat_firstpeak.append(C.mean(axis=1)[np.min(mask)])
        spec_heat_lastpeak.append(C.mean(axis=1)[np.max(mask)])
        spec_heat_maxpeak.append(C.mean(axis=1).max())

        mask_max_list.append(mask.max())
        # critical_entropy.append(entropy.mean(axis=1)[np.min(mask)])
        spec_heat_c.append(C.mean(axis=1)[mask.min()])


    return (mask_list, tau_firstpeak, tau_lastpeak, mask_max_list, tau_maxpeak,
            critical_entropy_firstpeak, critical_entropy_lastpeak, critical_entropy_maxpeak,
            spec_heat_firstpeak, spec_heat_lastpeak, spec_heat_maxpeak, entropy_save, C_save)


def get_tau_max_specific_heat_avareges(spectrum, tau_range, r_list):
    tau_error_peak_list = []

    tau_firstpeak = []
    tau_lastpeak = []
    tau_maxpeak = []

    critical_entropy_firstpeak = []
    critical_entropy_lastpeak = []
    critical_entropy_maxpeak = []

    spec_heat_firstpeak = []
    spec_heat_lastpeak = []
    spec_heat_maxpeak = []

    entropy_save = []
    C_save = []

    mask_max_list = []
    # critical_entropy = []
    spec_heat_c = []
    for key in r_list:
    # entropy, Z, _ = von_neumann_entropy_numpy_isotherm_isobar(tau_range=tau_range,
        #                                                           lambd=spectrum[:, r_ == key].squeeze(),
        #                                                           volume=volume[:, r_ == key].squeeze())
        entropy, Z, _ = von_neumann_entropy_numpy(tau_range=tau_range, lambd=spectrum[:, r_list == key].squeeze())
        entropy = entropy / np.log(spectrum.shape[-1])
        entropy = np.clip(a=entropy, a_min=0, a_max=1e15)
        entropy_save.append(entropy)

        C = np.log(spectrum.shape[-1])*np.clip(a_min=0, a_max=1e15, a=specific_heat(spec_ent=entropy, tau_range=tau_range, batch_dim=1))
        C_save.append(C)

        eps = 1e-18
        atol = 5e-10

        mask = find_peaks_indices(x_array=tau_range,
                                  y_array=C.mean(axis=1), #-np.gradient(entropy, np.log(tau_range), axis=0).mean(axis=1),
                                  eps=eps, atol=atol)

        tau_firstpeak.append(tau_range[mask].min())
        tau_lastpeak.append(tau_range[mask].max())
        tau_maxpeak.append(tau_range[np.argmax(C.mean(axis=1)[mask])])

        critical_entropy_firstpeak.append(entropy.mean(axis=1)[np.min(mask)])
        critical_entropy_lastpeak.append(entropy.mean(axis=1)[np.max(mask)])
        critical_entropy_maxpeak.append(entropy.mean(axis=1)[np.argmax(C.mean(axis=1)[mask])])

        spec_heat_firstpeak.append(C.mean(axis=1)[np.min(mask)])
        spec_heat_lastpeak.append(C.mean(axis=1)[np.max(mask)])
        spec_heat_maxpeak.append(C.mean(axis=1).max())

        mask_max_list.append(mask.max())
        # critical_entropy.append(entropy.mean(axis=1)[np.min(mask)])
        spec_heat_c.append(C.mean(axis=1)[mask.min()])

        # plot_von_neumann_ent(von_neumann_ent=entropy, tau_range=tau_range,
        #                      # tau_of_lastpeak_list=1/np.sort(spectrum, axis=1)[:, 1],
        #                      tau_star_list=[tau_range[mask].max()],
        #                      spec_heat=C,
        #                      ylim_ax2=(-.01, 2.),
        #                      labely=r'$\mathbf{S_{\tau}}$',
        #                      take_average=True,
        #                      legend_title='r={:.2f}'.format(key),
        #                      show=args.verbose,
        #                      fig_format=f'{args.fig_format}',
        #                      save_name='{:s}ent_r{:05.2f}'.format(save_fig_spectrum_vne, key))
        # plt.close()

        # a=0
        # F = -np.log(Z)
        # plot_von_neumann_ent(von_neumann_ent_to_plot=F, tau_range=tau_range, labely=r'$\mathbf{F_{\tau}}$',
        #                      legend_title=f'r={str(key)}', show=True, fig_format=f'{args.fig_format}',
        #                      save_name='{:s}free_ener_r{:s}'.format(save_fig_vne, str(key)))
        # plt.close()
    # tau_star = np.array(unroll_nested_list(tau_of_lastpeak_list))

    return (tau_firstpeak, tau_lastpeak, mask_max_list, tau_maxpeak,
            critical_entropy_firstpeak, critical_entropy_lastpeak, critical_entropy_maxpeak,
            spec_heat_firstpeak, spec_heat_lastpeak, spec_heat_maxpeak, entropy_save, C_save)
