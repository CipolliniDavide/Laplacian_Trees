import sys

import pandas as pd
import argparse
from argparse import RawTextHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

import torch
import networkx as nx


# from degree_figures import plot_table_of_degree_distribution
from helpers.utils import unroll_nested_list, ensure_dir
from helpers.visual_utils import set_ticks_label, set_legend, create_colorbar, get_set_larger_ticks_and_labels, scientific_notation
from helpers_dataset.helpers_dataset import (order_files_by_r, read_csv_convert_to_dataframe, create_dataset_Portegys,
                                             load_key_by_, check_dataset,
                                             plot_algorithmic_accuracy, plot_ntw_properties, filter_file_list)
from utils_spectral_entropy.spectral_entropy_numpy import (von_neumann_entropy_numpy,
                                                           specific_heat, entropy_var_ensemble)
from utils_spectral_entropy.utils import find_peaks_indices
from utils_spectral_entropy.thermo_efficiency import thermo_trajectory, thermo_efficiency_by_key
from utils_spectral_entropy.make_plots import (plot_von_neumann_ent, plot_thermo_trajectory, plot_thermo_trajectory_separate,
                                               plot_eta_curves, plot_curves, plot_quantity_along_transition,
                                               plot_eta_dS_dF, plot_surface, plot_bothefficiency, plot_tau_vs_r)
from utils_spectral_entropy.ensamble_average import ensamble_avarage, ensamble_average_batch
from utils_degree.degree_distribution import compare_bar_plot
from utils_spectrum.plot_eigen import plot_eigen, plot_propagator, plot_propagator_eigenvalues
from utils_spectrum.log_log_distribution import plot_log_log_distribution
from utils_spectral_entropy.get_tau_max_specific_heat import get_tau_max_specific_heat
import matplotlib.ticker as ticker


rng = np.random.default_rng(1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate the log-spaced array
log_min = -2  # Start value (10^-3)
log_max = 6  # Stop value (10^4)
num_points = 1000  # Number of points
tau_range = np.logspace(log_min, log_max, num=num_points)


if  __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-ds', '--dataset_dir',
                        # default=f'{os.getcwd()}/Dataset/boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=3566897019',
                        # default=f'{os.getcwd()}/Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=64186134',
                        default=f'{os.getcwd()}/Dataset/boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=500,500-s=1',
                        # default=f'{os.getcwd()}/Dataset/boom-nist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1',
                        # default=f'{os.getcwd()}/LaplacianTree/Dataset/boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1',
                        # default=f'{os.getcwd()}//Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1',
                        type=str, help='Name of dataset_dir.')
    # parser.add_argument('-s_dir', '--save_dir', default=f'{os.getcwd()}/Output/PortegysTree', type=str, help='Name of dataset_dir.')
    parser.add_argument('-fig_form', '--fig_format', default='pdf', type=str, help='')
    parser.add_argument('-sp', '--spectrum_type', default='spectrum', type=str, help='')
    parser.add_argument("-v", "--verbose", action="store_true", help="Suppresses all output if False")
    parser.add_argument("-maxit", '--max_iter', default=50, type=int, help='Maximum number of iterations to include in the dataset')
    args = parser.parse_args()
    print(args)
    # print(device)

    root = os.getcwd()

    # Define save dirs
    args.dataset_dir = args.dataset_dir + '/'
    save_fold = args.dataset_dir.replace('Dataset', 'Output/{:s}/'.format(args.fig_format))
    save_fig_spectrum = f'{save_fold}{args.spectrum_type}/'
    save_fig_spectrum_vne = f'{save_fold}{args.spectrum_type}/vne/'
    os.makedirs(save_fig_spectrum, exist_ok=True)
    os.makedirs(save_fig_spectrum_vne, exist_ok=True)

    # Load networks:
    load_path = f'{args.dataset_dir}/output/'
    file_list = filter_file_list(sorted(glob('{:s}/*.tree'.format(load_path))), max_value=args.max_iter)
    dict_files = order_files_by_r(file_list=file_list)

    print(f'\nSave fold:\n\t{save_fold}')


    # Load dataset containing algorithm performance
    dataset_name = args.dataset_dir.split('boom-')[1].split('-')[0]
    data = read_csv_convert_to_dataframe(file_path=args.dataset_dir + 'dataset.csv', save_fold=f'{save_fold}/',
                                         figformat=args.fig_format,
                                         keep_r_=(0, 1))
    # show=True)

    plot_algorithmic_accuracy(df=data,
                              r_range=(0, .85),
                              save_dir=save_fold,
                              fig_name='{:s}_alg_efficiency'.format(dataset_name),
                              fig_format=args.fig_format,
                              fig_size=(18, 4),
                              x_ticks=np.arange(0.1, .9, .1),
                              valfmt_x="{x:.1f}",
                              x_tick_every=2,
                              show=args.verbose)
                              # show=True)

    # Gather tree-network dataset
    save_prop_dir = load_path.replace('output', 'prop')
    create_dataset_Portegys(save_fold_ds=save_fold,
                            dict_files=False,
                            save_fold_prop=save_prop_dir,
                            crt_properties=False)

    # Load and prepare dataset according to limit on R and independent iterations
    df_nw = pd.read_hdf(f'{save_fold}/dataset_nw.h5', key='data').sort_values(by=['R', 'iter'])
    # df_nw = df_nw[(df_nw['R'] < .9) & (df_nw['iter'] < args.max_iter)]
    df_nw = df_nw[(df_nw['R'] <= 2) & (df_nw['iter'] < args.max_iter)]

    r_ = np.unique(df_nw['R'].values)
    # Check that R is equally represented
    check_dataset(df=df_nw, max_iter=args.max_iter)

    ############################# Get basic properties #####################
    df_temp = df_nw[df_nw['n_nodes'] < 2000]
    n_nodes = np.unique(df_nw['n_nodes'].values)[0]
    cc = np.unique(df_nw['connected_components'].values)

    plot_ntw_properties(r_=r_, df=df_nw, save_dir=save_fold, show=args.verbose, fig_format=args.fig_format)

    ############################################ Spectrum ##############################################################
    index_to_plot = [0, 3, 6, 13, -1]
    for key in ['spectrum', 'spectrum_rw']: #, 'eigvec', 'eigvec_rw']:
    # for key in ['spectrum']:  # , 'eigvec', 'eigvec_rw']:
        temp = np.clip(load_key_by_(key_name=key, max_iter=args.max_iter,
                                    load_dir=save_prop_dir, r_range=(r_.min(), r_.max())), a_min=0, a_max=np.inf).mean(axis=0)
        if 'rw' in key: ref_c = False
        else: ref_c = True
        plot_eigen(curve_labels=r_,
                   eigen_list_of_list=temp,
                   eigenwhat='eigenvalues',
                   index_to_plot=index_to_plot,
                   save_path=save_fold,
                   figname='laplacian_{:s}'.format(key),
                   figsize=(6, 5),
                   # title='laplacian {:s}'.format(key),
                   fig_format=args.fig_format,
                   legend_ncol=2,
                   show=args.verbose,
                   reference_curve=ref_c)
        plt.close('all')

    save_pow_law = save_fold+'spec_pow_law/'
    os.makedirs(name=save_pow_law, exist_ok=True)
    for ind, _ in enumerate(r_):
        temp = np.clip(load_key_by_(key_name='spectrum', max_iter=args.max_iter,
                                    load_dir=save_prop_dir, r_range=(r_.min(), r_.max())),
                       a_min=0, a_max=np.inf).mean(axis=0)
        plot_log_log_distribution(size_cc=[temp[ind]],
                                  N=[r_[ind]],
                                  x_lim_fit=(1e-3, .5),
                                  title_leg='r',
                                  save_dir=save_pow_law,
                                  fig_name=f'powlaw_{key}_r={r_[ind]:.2f}',
                                  fig_format=args.fig_format,
                                  figsize=(5, 4),
                                  cmap_='inferno',
                                  # y_lim=(1e-5, 10),
                                  legend_flag=False,
                                  make_fit_flag=True if r_[ind] > .4 else False,
                                  # show=True
                                  show=args.verbose
                                  )

    a=0
    # key = 'spectrum'
    # temp = load_key_by_(key_name='spectrum', load_dir=save_prop_dir, max_iter=args.max_iter)
    # lrho = np.exp(-np.multiply.outer(tau_range, temp)).mean(axis=1)  # + pressure * volume[:, np.newaxis]))
    #
    # for ind in index_to_plot:
    #     plot_propagator_eigenvalues(tau_range=tau_range, lrho=lrho[:, ind], save_dir=save_fold,
    #                                 title='r={:.2f},laplacian {:s}'.format(r_[ind], key), save_name='propagator_r={:.2f}'.format(r_[ind]))

    ####################################### Degree distribution #######################################################
    # save_fig_degree_distr = save_fold +'deg_dist/'
    # ensure_dir(save_fig_degree_distr)
    # for key in list(dict_files.keys()):
    #     compare_bar_plot(deg_list_of_list=df_nw[df_nw['R'] == key]['degrees'], save_fold=save_fig_degree_distr,
    #                      legend_title='{:.2f}'.format(key), binning_type='log', show_erdos_renyi=True,
    #                      figsize=(6, 5), labelx='k', labely='P(k)', show=False, fig_format=f'{args.fig_format}')
    #     plt.close()

    ######################################### Ensemble averages, and Var: S, D, L #########################################################
    spectrum = load_key_by_(max_iter=args.max_iter, key_name=args.spectrum_type, load_dir=save_prop_dir, r_range=(r_.min(), r_.max()))
    volume = load_key_by_(max_iter=args.max_iter, key_name='diameter', load_dir=save_prop_dir, r_range=(r_.min(), r_.max()))
    degree = load_key_by_(max_iter=args.max_iter, key_name='degrees', load_dir=save_prop_dir, r_range=(r_.min(), r_.max()))

    # Entropy S
    entropy_ensemble = np.zeros(shape=(args.max_iter, len(r_),) + tau_range.shape)
    entropy_var = np.zeros(shape=(args.max_iter, len(r_),) + tau_range.shape)
    chi = np.zeros(shape=(args.max_iter, len(r_),) + tau_range.shape)
    Z_arr = np.zeros(shape=(args.max_iter, len(r_),) + tau_range.shape)

    for r_ind in range(len(r_)):
        # entropy, Z, _ = von_neumann_entropy_numpy(tau_range=tau_range, lambd=spectrum[:, r_ind].squeeze())
        for it_index, it in enumerate(range(args.max_iter)):
            ent, var_ent, Z = entropy_var_ensemble(lambd=spectrum[it_index, r_ind], beta_range=tau_range)
            entropy_ensemble[it_index, r_ind] = ent
            entropy_var[it_index, r_ind] = var_ent
            Z_arr[it_index, r_ind] = Z
            chi[it_index, r_ind] = np.clip(a_min=0, a_max=1e15, a=specific_heat(spec_ent=ent, tau_range=tau_range, batch_dim=1))
    (mask_of_peaks_list, tau_firstpeak, tau_lastpeak, mask_max_list, tau_maxpeak,
     critical_entropy_firstpeak, critical_entropy_lastpeak, critical_entropy_maxpeak,
     spec_heat_firstpeak, spec_heat_lastpeak, spec_heat_maxpeak, entropy_save, C_save) = get_tau_max_specific_heat(spectrum=spectrum,
                                                                                                                   r_list=r_,
                                                                                                                   tau_range=tau_range
                                                                                                                   )

    # Loop over each unique eta value to compute the mean for entropy and specific heat
    for idx, r in enumerate(r_):
        plot_von_neumann_ent(von_neumann_ent=entropy_ensemble[:, idx, :].T/entropy_ensemble.max(),
                             valfmt_spec_entropy="{x:.0f}",
                             tau_range=tau_range,
                             # tau_of_lastpeak_list=1/np.sort(spectrum, axis=1)[:, 1],
                             tau_star_list=[tau_range[mask_of_peaks_list[idx]].max(),
                                            tau_range[mask_of_peaks_list[idx]].min()],
                             spec_heat=chi[:, idx, :].T,
                             ylim_ax2=(-.01, 2.),
                             labely=r'$\mathbf{S}$',
                             C_label='$\mathbf{C}$',
                             take_average=True,
                             legend_title='r={:.2f}'.format(r),
                             show=args.verbose,
                             fig_format=f'{args.fig_format}',
                             fontsize_ticks=26,
                             fontsize_labels=40,
                             fontsize_legend_title=35,
                             ticks_spec_heat=[.5, .66, 1],
                             x_ticks=[10, 10 ** 5], #, 10 ** 6],
                             valfmt_spec_heat="{x:.2f}",
                             grid_flag=True,
                             figsize=(8, 5),
                             save_name='{:s}ent_eta{:05.2f}'.format(save_fig_spectrum_vne, r))
        plt.close()

    entropy_between_peaks = np.zeros((args.max_iter, len(r_)))
    for ind_r, _ in enumerate(r_):
        entropy_between_peaks[:, ind_r] = entropy_ensemble[:, ind_r, min(mask_of_peaks_list[ind_r])] - entropy_ensemble[:, ind_r, max(mask_of_peaks_list[ind_r])]


    # plot_quantity_along_transition(r_=r_,
    #                                y_lim=(0, n_nodes*entropy_between_peaks.var(0).max()),
    #                                quantity=n_nodes*entropy_between_peaks.var(0),
    #                                figsize=(8, 5),
    #                                # x_ticks=r_[::6],
    #                                x_ticks=[0, .2, .4, 1, 1.5, 2],
    #                                fontsize_labels=30,
    #                                fontsize_ticks=25,
    #                                color='black',
    #                                marker='o',
    #                                fig_name='Varbetween_critical_entropy',
    #                                fig_format=args.fig_format,
    #                                ylabel=r'$\mathbf{N \cdot Var[S(\tau^{(1)}_{peak}) - S(\tau^{(2)}_{peak})]}$',
    #                                save_dir=save_fig_spectrum)

    x_ticks_for_plots = [.1, .4, .8, 1.2, 2.]
    plot_quantity_along_transition(r_=r_,
                                   quantity=np.array(critical_entropy_firstpeak) - np.array(critical_entropy_lastpeak),
                                   std_quantity=n_nodes * entropy_between_peaks.var(0),
                                   ylabel_std_quantity=r"$\mathbf{N \cdot Var}$",
                                   figsize=(8, 5),
                                   # x_ticks=r_[::6],
                                   x_ticks=x_ticks_for_plots,
                                   minor_ticks=[1],
                                   color='black',
                                   marker='o',
                                   fontsize_ticks=30,
                                   fontsize_labels=32,
                                   # fontsize_legend_title=30,
                                   fig_name='between_critical_entropy',
                                   fig_format=args.fig_format,
                                   ylabel=r'$\mathbf{S^{(1)}_{peak} - S^{(2)}_{peak}}$',
                                   save_dir=save_fig_spectrum,
                                   )

    plot_quantity_along_transition(r_=r_,
                                   quantity=critical_entropy_lastpeak,
                                   figsize=(8, 5),
                                   x_ticks=r_[::4],
                                   color='green',
                                   marker='o',
                                   fig_name='critical_entropy_lastpeak',
                                   fig_format=args.fig_format, ylabel=r'$\mathbf{S(\tau^{(2)}_{peak})}$',
                                   save_dir=save_fig_spectrum)
    plot_quantity_along_transition(r_=r_, quantity=critical_entropy_firstpeak, figsize=(8, 5),
                                   x_ticks=r_[::4],
                                   color='blue',
                                   marker='o',
                                   fig_name='critical_entropy_firstpeak',
                                   fig_format=args.fig_format, ylabel=r'$\mathbf{S(\tau^{(1)}_{peak})}$',
                                   save_dir=save_fig_spectrum)

    a=0
    # tau_lim = (1e0, 5e0)
    # tau_index = np.where((tau_range > tau_lim[0]) & (tau_range < tau_lim[1]))[0][::20]
    # plt.plot(r_, entropy_ensemble[..., tau_index].mean(axis=0))
    # plt.show()
    tau_range_reshaped = tau_range[np.newaxis, np.newaxis, :]
    F = - np.log(Z_arr)/tau_range_reshaped

    fig, ax = plt.subplots(figsize=(8, 6), ncols=1, nrows=1, layout='tight')
    for r_ind, r in enumerate(r_):
        if r_ind % 2 == 0:
            ax.scatter(entropy_save[r_ind].mean(1), F.mean(0)[r_ind], alpha=0.4, label='{:.2f}'.format(r))
    # ax.set_yscale('log')
    # set_ticks_label(ax=ax, ax_type='x', ax_label='S', data=[1, 0], valfmt="{x:.2f}")
    # set_ticks_label(ax=ax, ax_type='y', ax_label='F', data=F, valfmt="{x:.2f}")
    set_legend(ax=ax, title='r', ncol=3)
    plt.savefig(save_fig_spectrum + 'F_vs_S{:s}'.format(args.fig_format))
    # plt.show()

    plot_thermo_trajectory_separate(tau_range=tau_range,
                                    r=r_,
                                    # vmax=-1.3,
                                    networks_eta=F.mean(axis=0).T,
                                    fig_name='free_energy',
                                    add_yticks_heat=[np.log10(10), np.log10(100)],
                                    y_label='F',
                                    save_dir=save_fig_spectrum,
                                    cbar_ticks=[0, F.max()],
                                    ticks_size=30,
                                    label_size=40,
                                    figsize=(8, 5),
                                    title='',
                                    num_xticks=len(r_) // 2,
                                    x_ticks=r_[::4],
                                    tau_lim=(.01, 1e6),
                                    number_of_curves=10,
                                    heat_num_yticks=2,
                                    fig_format=args.fig_format,
                                    show=args.verbose
                                    # show=True
                                    )

    plot_thermo_trajectory_separate(tau_range=tau_range,
                                    r=r_,
                                    # vmax=-1.3,
                                    networks_eta=F.var(axis=0).T,
                                    fig_name='free_energy_var',
                                    add_yticks_heat=[np.log10(10), np.log10(100)],
                                    y_label='Var(F)',
                                    save_dir=save_fig_spectrum,
                                    figsize=(8, 5),
                                    title='',
                                    num_xticks=len(r_) // 2,
                                    tau_lim=(.01, 1e6),
                                    number_of_curves=10,
                                    heat_num_yticks=2,
                                    x_ticks=r_[::4],
                                    valfmt_cbar="{x:.3f}",
                                    cbar_ticks=[0, F.var(axis=0).max()],
                                    ticks_size=30,
                                    label_size=40,
                                    fig_format=args.fig_format,
                                    show=args.verbose
                                    # show=True

                                    )

    plot_thermo_trajectory_separate(tau_range=tau_range,
                                    r=r_,
                                    vmax=1.3,
                                    networks_eta=chi.mean(axis=0).T,
                                    fig_name='specific_heat',
                                    y_label='C',
                                    save_dir=save_fig_spectrum,
                                    figsize=(8, 5),
                                    title='',
                                    num_xticks=len(r_) // 2,
                                    # tau_lim=(.1, 1e6),
                                    number_of_curves=10,
                                    tau_lim=(.01, 1e6),
                                    x_ticks=x_ticks_for_plots,
                                    cbar_ticks=[0, 1.3, .66],
                                    heat_num_yticks=1,
                                    add_yticks_heat=[1., 5],
                                    # add_yticks_heat=[np.log10(10), np.log10(100), np.log10(1000), np.log10(10000),
                                    #                  np.log10(100000)],
                                    ticks_size=20,
                                    label_size=30,
                                    fig_format=args.fig_format,
                                    show=args.verbose
                                    # show=True
                                    )

    plot_thermo_trajectory_separate(tau_range=tau_range,
                                    r=r_,
                                    vmax=1.,
                                    networks_eta=entropy_ensemble.mean(axis=0).T,
                                    fig_name='ensemble_S',
                                    add_yticks_heat=[np.log10(10), np.log10(100)],
                                    y_label='S',
                                    save_dir=save_fig_spectrum,
                                    figsize=(8, 5),
                                    title='',
                                    num_xticks=len(r_) // 2,
                                    x_ticks=r_[::4],
                                    tau_lim=(.01, 1e6),
                                    number_of_curves=10,
                                    heat_num_yticks=2,
                                    fig_format=args.fig_format,
                                    show=args.verbose
                                    # show=True
                                    )

    plot_thermo_trajectory_separate(tau_range=tau_range,
                                    r=r_,
                                    vmax=1.3,
                                    networks_eta=entropy_var.mean(axis=0).T,
                                    fig_name='ensemble_varS',
                                    add_yticks_heat=[np.log10(10), np.log10(100)],
                                    y_label='Var(S)',
                                    save_dir=save_fig_spectrum,
                                    figsize=(8, 5),
                                    title='',
                                    num_xticks=len(r_) // 2,
                                    x_ticks=r_[::4],
                                    tau_lim=(.01, 1e6),
                                    number_of_curves=10,
                                    heat_num_yticks=2,
                                    fig_format=args.fig_format,
                                    show=args.verbose
                                    # show=True
                                    )

    # plt.imshow(entropy_var.mean(axis=0), aspect='auto')
    # plt.yticks(np.log(tau_range))
    # plt.xticks(1/np.log(r_))
    # plt.show()
    # # plt.plot(q2_ens)
    # plt.show()

    # D, L
    for data, ylab, ylab_var, fig_name in [
                                           # (degree, r'$\mathbf{<\hat{D}>}$', '$\mathbf{Var(\hat{D})}$', 'degree'),
                                           (spectrum, r'$\mathbf{<\hat{L}>}$', '$\mathbf{Var(\hat{L})}$', 'U'),
                                            ]:
        save_fig_q = f'{save_fold}{fig_name}/'
        ensure_dir(save_fig_q)

        # Compute quantity ensemble avg
        q_ensemble = np.zeros(shape=(args.max_iter, len(r_),)+tau_range.shape)
        q_var_ensemble = np.zeros(shape=(args.max_iter, len(r_),) + tau_range.shape)

        for r_ind in range(len(r_)):
            for it_index, it in enumerate(range(args.max_iter)):
                q_ensemble[it_index, r_ind], _ = ensamble_avarage(order_parameter=data[it_index, r_ind],
                                                                    lambd=spectrum[it_index, r_ind],
                                                                    beta_range=tau_range)
                q2_ens, _ = ensamble_avarage(order_parameter=data[it_index, r_ind]**2,
                                             lambd=spectrum[it_index, r_ind],
                                             beta_range=tau_range)
                q_var_ensemble[it_index, r_ind] = q2_ens - q_ensemble[it_index, r_ind] ** 2

        if '{<\hat{L}>}' in ylab:
            U = q_ensemble
        q_mean_ens_avg = q_ensemble.mean(axis=0)
        q_var_ens_avg = q_var_ensemble.mean(axis=0)

        fig, ax = plt.subplots()
        ax.plot(r_, q_var_ens_avg.max(1))
        plt.savefig(save_fig_q + f'{fig_name}_max_across_tau.png')
        plt.close()

        # plt.imshow(deg_ensamble.std(axis=0), aspect='auto')
        # plt.show()
        plot_thermo_trajectory_separate(tau_range=tau_range,
                                        r=r_,
                                        networks_eta=q_mean_ens_avg.T,
                                        fig_name='ensemble_' + fig_name,
                                        y_label=ylab,
                                        save_dir=save_fig_q,
                                        figsize=(8, 5),
                                        title='',
                                        heat_num_yticks=2,
                                        add_yticks_heat=[np.log10(10), np.log10(100)],
                                        num_xticks=len(r_) // 2,
                                        x_ticks=r_[::4],
                                        tau_lim=(.01, 1e6),
                                        # tau_lim=(1, 1e6),
                                        number_of_curves=10,
                                        fig_format=args.fig_format,
                                        # show=True
                                        show=args.verbose
                                        )
        plot_thermo_trajectory_separate(tau_range=tau_range,
                                        r=r_,
                                        # vmax=q_var_ens_avg.T.max(),
                                        # networks_eta=deg_ensamble.std(axis=0).T ** 2,
                                        networks_eta=q_var_ens_avg.T,
                                        fig_name='ensemble_Var'+fig_name,
                                        y_label=ylab_var,
                                        heat_num_yticks=2,
                                        add_yticks_heat=[np.log10(10), np.log10(100)],
                                        save_dir=save_fig_q,
                                        figsize=(8, 5),
                                        title='',
                                        num_xticks=len(r_) // 2,
                                        x_ticks=r_[::4],
                                        # tau_lim=(.1, 1e6),
                                        tau_lim=(tau_range[0], 1e6),
                                        number_of_curves=10,
                                        fig_format=args.fig_format,
                                        show=args.verbose
                                        # show=True
                                        )
        tau_lim_deg = (7., 1e2)
        plot_thermo_trajectory_separate(tau_range=tau_range,
                                        r=r_,
                                        networks_eta=q_mean_ens_avg.T,
                                        fig_name='ensemble_{:s}_lim={:.1f},{:.1f}'.format(fig_name, tau_lim_deg[0], tau_lim_deg[1]),
                                        y_label=ylab,
                                        save_dir=save_fig_q,
                                        figsize=(8, 5),
                                        heat_num_yticks=2,
                                        add_yticks_heat=[np.log10(10), np.log10(100)],
                                        title='',
                                        num_xticks=len(r_) // 2,
                                        x_ticks=r_[::4],
                                        tau_lim=tau_lim_deg,
                                        number_of_curves=10,
                                        fig_format=args.fig_format,
                                        show=args.verbose
                                        )
        plot_thermo_trajectory_separate(tau_range=tau_range,
                                        r=r_,
                                        # networks_eta=deg_ensamble.std(axis=0).T**2,
                                        networks_eta=q_var_ens_avg.T,
                                        fig_name='ensemble_Var_{:s}_lim={:.1f},{:.1f}'.format(fig_name, tau_lim_deg[0], tau_lim_deg[1]),
                                        y_label=ylab_var,
                                        save_dir=save_fig_q,
                                        figsize=(8, 5),
                                        heat_num_yticks=2,
                                        add_yticks_heat=[np.log10(10), np.log10(100)],
                                        title='',
                                        num_xticks=len(r_) // 2,
                                        x_ticks=r_[::4],
                                        tau_lim=tau_lim_deg,
                                        number_of_curves=10,
                                        fig_format=args.fig_format,
                                        show=args.verbose
                                        )

    ########################################################################################################

    plot_tau_vs_r(r_=r_, tau_lastpeak=np.array(tau_lastpeak), tau_firstpeak=np.array(tau_firstpeak), spectrum=spectrum,
                  save_dir=save_fig_spectrum, fig_format=args.fig_format, y_lim=(.3, 1e5),
                  show=args.verbose)
                  # show=True)
    a=0

    from scipy.signal import savgol_filter
    # Phase plane :
    tau_lastpeak = np.array(tau_lastpeak)
    y_lim = (.3, 1e6)
    box_pts=6
    # box = np.ones(box_pts) / box_pts
    # tau_firstpeak_ph = np.convolve(tau_firstpeak, box, mode='same')
    # tau_lastpeak_ph = np.convolve(tau_lastpeak, box, mode='same')
    # tau_lastpeak_ph = np.copy(tau_lastpeak)
    # indexes = np.where(tau_lastpeak > tau_firstpeak)[0]
    # tau_lastpeak_ph[indexes] = savgol_filter(tau_lastpeak[indexes], polyorder=2, window_length=box_pts)

    # tau_firstpeak_ph = savgol_filter(tau_firstpeak, polyorder=3, window_length=box_pts)
    tau_lastpeak_ph = tau_lastpeak
    tau_firstpeak_ph = tau_firstpeak

    # Phase Diagram

    figsize = (8, 6)
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)  # Change layout to tight_layout
    ax.fill_between(r_, y1=tau_firstpeak_ph, y2=tau_lastpeak_ph, color='mediumaquamarine', alpha=.6)
    ax.fill_between(r_, y1=0, y2=tau_firstpeak_ph, color='darkslategrey', alpha=.5)
    ax.fill_between(r_, y1=y_lim[1], y2=tau_lastpeak_ph, color='lightyellow', alpha=.5)

    ax.scatter(r_, np.array(tau_lastpeak_ph), label=r'$\tau^{(2)}_{peak}$', c='green', alpha=.6, lw=5)
    ax.scatter(r_, np.array(tau_firstpeak_ph), label=r'$\tau^{(1)}_{peak}$', c='blue', alpha=.4, lw=5)

    ax.set_yscale('log')
    label_size = 30
    ticks_size = 25
    # ax.set_xscale('log')
    set_ticks_label(ax=ax, ax_type='y',
                    # data=np.log10(tau_range),
                    data=tau_range,
                    num=1,
                    add_ticks=[10, 1e5],
                    # valfmt=valfmt_y,
                    ticks=None,
                    only_ticks=False,
                    tick_lab=None,
                    fontdict_label={'weight': 'bold', 'size': label_size, 'color': 'black'},
                    fontdict_ticks_label={'weight': 'bold', 'size': ticks_size},
                    label_pad=.1,
                    ax_label=r'$\mathbf{\tau}$',
                    scale=None,
                    )
    # Customize tick parameters for larger and longer ticks on both axes
    ax.tick_params(axis='both', which='major', length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', length=5, width=1.5)  # Minor ticks
    ax.get_yaxis().set_major_formatter(ticker.LogFormatterMathtext(base=10))  # Use scalar format for readability

    set_ticks_label(ax=ax, ax_type='x',
                    # data=np.log10(tau_range),
                    data=r_,
                    valfmt="{x:.1f}",
                    ticks=x_ticks_for_plots,
                    only_ticks=False, tick_lab=None,
                    fontdict_label={'weight': 'bold', 'size': label_size, 'color': 'black'},
                    fontdict_ticks_label={'weight': 'bold', 'size': ticks_size},
                    label_pad=.1,
                    ax_label=r'$\mathbf{r}$',
                    scale=None,
                    add_ticks=[])

    # ax.set_xlabel('r')
    # ax.set_ylabel(r'$\mathbf{\tau}$')
    if y_lim:
        ax.set_ylim(10**-1, y_lim[1])
    ax.set_xlim(left=r_[0], right=r_[-1])
    # get_set_larger_ticks_and_labels(ax=ax, num_ticks_x=10)
    # set_legend(ax=ax)
    plt.savefig(save_fig_spectrum + 'phase_diagram.png', dpi=300)
    # plt.show()

    ####################################### Plot surface of S, C, U ####################################################

    # plot_surface(z=F.mean(axis=0).T,
    #              r=r_,
    #              tau_range=tau_range,
    #              # z_lastpeak=z_lastpeak, z_firstpeak=z_firstpeak,
    #              # z_curve1=np.array(critical_entropy_lastpeak),  # +1e-2,
    #              # z_curve2=np.array(critical_entropy_firstpeak),  # +1e-2,
    #              tau_lastpeak=tau_lastpeak,
    #              tau_firstpeak=np.array(tau_firstpeak),
    #              fig_format=args.fig_format,
    #              save_dir=save_fold,
    #              # title=title,
    #              # z_lim=z_lim,
    #              z_label=F,
    #              # vmax=vmax,
    #              # alpha=alpha,
    #              # view_init=view_init,
    #              cmap='plasma',
    #              show=True
    #              )

    for z, label, title, view_init, label_overlayed_curve in [
                                    # (F.mean(axis=0), r'$\mathbf{F}$', 'free_energy', (30, 60)),
                                       (entropy_save, r'$\mathbf{S}$', 'entropy', (30, 50), (r'$S(\tau^{(1)}_{peak})$', r'$S(\tau^{(2)}_{peak})$')),
                                       (C_save, r'$\mathbf{C}$', 'specific_heat', (30, 40), (r'$C^{(1)}_{peak}$', r'$C^{(2)}_{peak}$'))
    ]:
        alpha = .65
        if title == 'entropy':
            z_lastpeak = np.array(critical_entropy_lastpeak)
            z_firstpeak = np.array(critical_entropy_firstpeak)
            z_lim = (None, None)
            vmax=1
        else:
            z_lim = (0, 3)
            z_lastpeak = np.array(spec_heat_lastpeak)
            z_firstpeak = np.array(spec_heat_firstpeak)
            vmax=1.5
        plot_surface(z=np.array(z).mean(axis=2).T,
                     r=r_,
                     tau_range=tau_range,
                     z_lastpeak=z_lastpeak, z_firstpeak=z_firstpeak,
                     z_curve1=np.array(critical_entropy_lastpeak), #+1e-2,
                     z_curve2=np.array(critical_entropy_firstpeak), #+1e-2,
                     tau_lastpeak=tau_lastpeak,
                     tau_firstpeak=np.array(tau_firstpeak),
                     fig_format=args.fig_format,
                     save_dir=save_fold,
                     title=title,
                     z_lim=z_lim,
                     z_label=label,
                     vmax=vmax,
                     alpha=alpha,
                     view_init=view_init,
                     cmap='plasma',
                     label_overlayed_curve=label_overlayed_curve,
                     show=args.verbose
                     # show=True
        )

    # critical_index = np.where(np.logical_and(tau_range >= 1e2, tau_range <= 5e2))[0]
    # plot_eta_dS_dF(eta_mean=eta_mean, dF_mean=dF_mean, dS_mean=dS_mean, r_list=r_list,
    #                save_dir=save_fold, tau_index=critical_index, tau_range=tau_range,
    #                top_ylim=1,
    #                legend_loc=1,
    #                show=True)


    ######### Plot critical entropy and, specific heat, ...
    for title, critical_entropy, critical_spec_heat, critical_tau in \
            [('first_peak', critical_entropy_firstpeak, spec_heat_firstpeak, tau_firstpeak),
            ('last_peak', critical_entropy_lastpeak, spec_heat_lastpeak, tau_lastpeak),
            ('max_peal', critical_entropy_maxpeak, spec_heat_maxpeak, tau_maxpeak)]:

            fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 16), ncols=1, nrows=3, sharex=True)
            ax1.scatter(r_, np.array(critical_entropy))
            ax2.scatter(r_, np.array(critical_spec_heat))
            ax3.scatter(r_, np.array(critical_tau))
            ax3.set_yscale('log')
            set_ticks_label(ax=ax1, ax_type='y', ax_label=r'$\mathbf{S_c}$', data=np.array(critical_entropy), valfmt="{x:.2f}")
            set_ticks_label(ax=ax2, ax_type='y', ax_label=r'$\mathbf{C_c}$', data=critical_spec_heat, valfmt="{x:.2f}")
            ax3.set_ylabel(r'$\mathbf{\tau}$')
            get_set_larger_ticks_and_labels(ax=ax3)
            for ax in [ax1, ax2, ax3]:
                ax.grid()
                set_ticks_label(ax=ax, ax_type='x', ax_label='r', ticks=r_[::2], data=None, valfmt="{x:.2f}")
            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(f'{save_fold}/{title}_coexistence_curve.png')
            if args.verbose:
                plt.show()
            else:
                plt.close()


    fig, ax = plt.subplots(figsize=(6, 4), ncols=1, nrows=1, layout='tight')
    ax.scatter(r_, tau_firstpeak)
    set_ticks_label(ax=ax, ax_type='x', ax_label='r', ticks=r_[::2], data=None, valfmt="{x:.2f}")
    get_set_larger_ticks_and_labels(ax=ax)
    ax.set_ylabel(r'$\tau^{(min)}_{peak}$')
    # set_ticks_label(ax=ax, ax_type='y', ax_label=, ticks=tau_of_firstpeak, data=None, valfmt="{x:.2f}")
    if args.verbose:
        plt.show()
    else:
        plt.close()


    ######################################### Plot efficiency ##########################################################
    # Compute ensamble efficiency, dS and dF
    networks_eta_avg, networks_eta_std, \
        dS_avg, dS_std, \
        dF_avg, dF_std = thermo_efficiency_by_key(tau_range=tau_range, spectrum=spectrum)

    eta_across_first_peak = []
    eta_across_last_peak = []
    for ind, _ in enumerate(r_):
        eta_across_first_peak.append(networks_eta_avg[mask_of_peaks_list[ind].min(), ind])
        eta_across_last_peak.append(networks_eta_avg[mask_of_peaks_list[ind].max(), ind])

    ind_to_plot_start = np.max([mask_temp.min() for mask_temp in mask_of_peaks_list])
    tau_of_legend = tau_range[ind_to_plot_start::5]
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, layout='tight', figsize=(10, 12))
    axes[0].scatter(r_, eta_across_first_peak)
    axes[1].scatter(r_, eta_across_last_peak)
    axes[2].plot(r_, networks_eta_avg[ind_to_plot_start::5].T)

    [ax.set_ylabel(lab) for ax, lab in zip(axes, [r"$\mathbf{\eta(\tau_{peak}^{(1)})}$", r"$\mathbf{\eta(\tau_{peak}^{(2)})}$", r"$\mathbf{\eta}$"])]
    [ax.grid() for ax in axes]
    # axes[1].set_xlabel(r"$\mathbf{r}$")
    set_ticks_label(ax=axes[1], ax_type='x', ax_label='r',
                    ticks=np.hstack((r_[::10], r_[np.argmax(eta_across_first_peak)])),
                    data=None, valfmt="{x:.2f}")
    # set_ticks_label(ax=axes[0], ax_type='y', ax_label='r',
    #                 ticks=np.hstack((r_[::10], r_[np.argmax(eta_across_first_peak)])),
    #                 data=None, valfmt="{x:.2f}")
    [get_set_larger_ticks_and_labels(ax=ax) for ax in axes]
    plt.savefig(save_fig_spectrum+'eta_across_transitions.{:s}'.format(args.fig_format))
    # plt.show()

    tau_lim = (.01, 5e2)
    tau_index = np.where((tau_range > tau_lim[0]) & (tau_range < tau_lim[1]))[0][::20]

    # for r_ind, r in enumerate(r_):
    # #     plt.plot(dS_avg[tau_index, r_ind], dF_avg[tau_index, r_ind], label='r={:.2f}'.format(r))
    #     for c in enumerate(tau_index):
    #         plt.plot(dS_avg[c, r_ind], dF_avg[c, r_ind], label=r)
    #     plt.legend()
    #     plt.show()
    # plt.imshow()

    plot_eta_dS_dF(eta_mean=networks_eta_avg.T, dF_mean=dF_avg.T,
                   dS_mean=dS_avg.T,
                   ylabels=[r'$\mathbf{\eta}$', r'$\mathbf{\delta S}$', r'$\mathbf{\delta F}$'],
                   x_ticks=r_[::4],
                   r_list=r_,
                   save_dir=save_fig_spectrum,
                   tau_index=tau_index,
                   tau_range=tau_range,
                   top_ylim=1,
                   legend_loc=1,
                   ncol_leg=2,
                   show=args.verbose)

    plot_eta_dS_dF(eta_mean=networks_eta_avg.T, dF_mean=dF_avg.T,
                   dS_mean=U.mean(axis=0),
                   ylabels=[r'$\mathbf{\eta}$', r'$\mathbf{U}$', r'$\mathbf{\delta F}$'],
                   x_ticks=r_[::4],
                   r_list=r_,
                   save_dir=save_fig_spectrum,
                   tau_index=tau_index,
                   tau_range=tau_range,
                   top_ylim=1,
                   legend_loc=1,
                   ncol_leg=2,
                   show=args.verbose,
                   fig_name='eta_U_dF')

    # plot_eta_dS_dF(eta_mean=networks_eta_std.T ** 2,
    #                dF_mean=dF_std.T ** 2,
    #                dS_mean=dS_std.T ** 2,
    #                # ylabels=[r'$\mathbf{\sigma_{\eta}}$', r'$\mathbf{\sigma_{dS}}$', r'$\mathbf{\sigma_{dF}}$'],
    #                ylabels=[r'$\mathbf{Var(\eta)}$', r'$\mathbf{Var(dS)}$', r'$\mathbf{Var(dF)}$'],
    #                r_list=r_,
    #                save_dir=save_fig_spectrum,
    #                tau_index=np.where((tau_range > tau_lim[0]) & (tau_range < tau_lim[1]))[0][::20],
    #                tau_range=tau_range,
    #                top_ylim=1,
    #                legend_loc=1,
    #                ncol_leg=2,
    #                fig_format=args.fig_format,
    #                # fig_name='std_eta_dS_dF',
    #                fig_name='var_eta_dS_dF',
    #                show=args.verbose)
    #

    # plt.plot(r_, networks_eta_std[np.where((tau_range>10) & (tau_range< 1e2))[0][::5]].T)
    # plt.show()

    plot_thermo_trajectory_separate(tau_range=tau_range,
                                    r=r_,
                                    networks_eta=dS_std ** 2,
                                    # fig_name='etastd',
                                    fig_name='new_var_{:s}'.format('$\mathbf{\delta S}$'),
                                    y_label=r'$\mathbf{Var(\delta S)}$',
                                    save_dir=f'{save_fig_spectrum}/',
                                    figsize=(8, 5),
                                    title='',
                                    x_ticks=r_[::4],
                                    # num_xticks=len(r_) // 2,
                                    tau_lim=(1, 1e6),
                                    # tau_lim=(.1, 1e6),
                                    # add_yticks_heat=[np.log10(10), np.log10(100), np.log10(1000), np.log10(10000), np.log10(100000)],
                                    add_yticks_heat=[np.log10(10), np.log10(100)],
                                    number_of_curves=10,
                                    fig_format=args.fig_format,
                                    show=args.verbose,
                                    heat_num_yticks=2,
                                    )

    for data, ylab, name in [
                             # (networks_eta_std, r'$\mathbf{Var(\eta)}$', 'eta'),
                             # (dF_std, r'$\mathbf{Var(\delta F)}$', '$\mathbf{\delta F}$'),
                             (dS_std, r'$\mathbf{Var(\delta S)}$', '$\mathbf{\delta S}$')]:
        plot_thermo_trajectory_separate(tau_range=tau_range,
                                        r=r_,
                                        networks_eta=data ** 2,
                                        # fig_name='etastd',
                                        fig_name='var_{:s}'.format(name),
                                        y_label=ylab,
                                        save_dir=f'{save_fig_spectrum}/',
                                        figsize=(8, 5),
                                        title='',
                                        x_ticks=r_[::4],
                                        # num_xticks=len(r_) // 2,
                                        tau_lim=(1, 1e6),
                                        # tau_lim=(.1, 1e6),
                                        # add_yticks_heat=[np.log10(10), np.log10(100), np.log10(1000), np.log10(10000), np.log10(100000)],
                                        add_yticks_heat=[np.log10(10), np.log10(100)],
                                        number_of_curves=10,
                                        fig_format=args.fig_format,
                                        show=args.verbose,
                                        heat_num_yticks=2,
                                        )

    # eta_mean = np.array([np.clip(a=networks_eta_avg, a_min=0, a_max=1)[m, i] for i, m in enumerate(mask_max_list)])
    # plt.plot(r_, eta_mean)
    # plt.show()
    # critical_index = np.where(np.logical_and(tau_range >= 1e1, tau_range <= 1e2))[0]
    # plot_bothefficiency(r_list=r_,
    #                     tau_range=tau_range,
    #                     eta_mean=np.clip(a=networks_eta_avg, a_min=0, a_max=1).T,
    #                     # eta_mean=np.clip(a=networks_eta_avg, a_min=0, a_max=1).T,
    #                     alg_eta_mean=data.groupby('R')['trade_off'].mean().values,
    #                     tau_index=np.array(mask_max_list),
    #                     show=args.verbose,
    #                     save_dir=save_fold)

    # if "mnist" in save_fold:
    #     tau_lim = (9, 1e2)
    # else:
    cmap_name = plt.get_cmap('inferno')
    slicedCM = cmap_name(np.linspace(0.1, .9, len(r_)))

    fig, ax = plt.subplots(figsize=(8, 6), ncols=1, nrows=1, layout='tight')
    for r_ind, r in enumerate(r_):
        if r_ind % 2 == 0:
            ax.scatter(dS_avg[:, r_ind], dF_avg[:, r_ind], c=slicedCM[r_ind], alpha=0.4, label='{:.2f}'.format(r))
    set_ticks_label(ax=ax, ax_type='x', ax_label='dS', data=dS_avg, valfmt="{x:.2f}")
    set_ticks_label(ax=ax, ax_type='y', ax_label='dF', data=dF_avg, valfmt="{x:.2f}")
    set_legend(ax=ax, title='r', ncol=3)
    plt.savefig(save_fig_spectrum+'dF_vs_dS{:s}'.format(args.fig_format))
    # plt.show()

    tau_mask = np.where((tau_range >= 1) & (tau_range <= 1000))[0]
    tau_mask = tau_mask[::len(tau_mask) // (9 - 1)]
    cmap_name = plt.get_cmap('inferno')
    slicedCM = cmap_name(np.linspace(0.1, .9, len(tau_mask)))
    fig, ax = plt.subplots(figsize=(8, 6), ncols=1, nrows=1, layout='tight')
    for t_ind, tau in enumerate(tau_range[tau_mask]):
        # if t_ind % 10 == 0:
        ax.plot(dS_avg[tau_mask][t_ind, :], dF_avg[tau_mask][t_ind, :],
                   c=slicedCM[t_ind], alpha=0.7, lw=5,
                   # label='{:.1e}'.format(tau)
                   label=scientific_notation(tau)
        )
    set_ticks_label(ax=ax, ax_type='x', ax_label='dS', data=dS_avg, valfmt="{x:.2f}")
    set_ticks_label(ax=ax, ax_type='y', ax_label='dF', data=dF_avg, valfmt="{x:.2f}")
    set_legend(ax=ax, title=r'$\mathbf{\tau}$', ncol=3)
    plt.savefig(save_fig_spectrum + 'dF_vs_dS_tau{:s}'.format(args.fig_format))
    # plt.show()

    tau_lim = (10, 9e1)
    plot_thermo_trajectory_separate(tau_range=tau_range, r=r_,
                                    networks_eta=np.clip(a=networks_eta_avg, a_min=0, a_max=1),
                                    save_dir=f'{save_fig_spectrum}/',
                                    figsize=(8, 5),
                                    title='',
                                    num_xticks=len(r_)//2,
                                    tau_lim=tau_lim,
                                    number_of_curves=8,
                                    fig_format=args.fig_format,
                                    show=args.verbose
                                    )
    plot_thermo_trajectory(tau_range=tau_range, r=r_,
                           # networks_eta=networks_eta_avg,
                           networks_eta=np.clip(a=networks_eta_avg, a_min=0, a_max=1),
                           save_dir=f'{save_fig_spectrum}/',
                           figsize=(8, 10),
                           title='',
                           num_xticks=len(r_)//2,
                           # tau_lim=(8e0, 1e4) # questo va molto bene per spectrum lapl ma il min dovrebbe essere 2 per corrispondere al primo picco
                           # tau_lim=(max(tau_stars), 1e4), # if using the method find_first_peak_x()
                           # tau_lim=(min(tau_stars)*1, max(tau_stars)*1)
                           # tau_lim=(10.1, 1e2),
                           tau_lim=tau_lim,
                           # tau_lim=(tau_star.min(), tau_star.max()),
                           number_of_curves=8,
                           # number_of_curves=9,
                           # tau_stars=tau_star
                           fig_format=args.fig_format,
                           show=args.verbose
                           )
    # print('\ntau* in: ({:.2e}, {:.2e})'.format(min(tau_of_lastpeak_list), max(tau_of_lastpeak_list)))

    ################################### Plot some extra curves for the thermodynamic efficiency #########################
    tau_selected = tau_range[np.array(mask_max_list)]
    plot_eta_curves(tau_range=tau_selected, r=r_, networks_eta=networks_eta_avg[mask_max_list])
    plt.savefig(f'{save_fig_spectrum}/curves_tau_star.{args.fig_format}')
    if args.verbose:
        plt.show()
    else:
        plt.close()


    min_lambda = np.array([np.sort(spectrum[:, i].squeeze(), axis=1)[:, 1] for i in range(spectrum.shape[1])])
    tau_diff_avg = (1 / min_lambda).mean(1)
    closest_indices = np.unique(np.abs(tau_range[:, None] - tau_diff_avg).argmin(axis=0))
    plot_eta_curves(r=r_,
                    tau_range=tau_range,
                    tau_stars=tau_range[closest_indices],
                    networks_eta=networks_eta_avg,
                    number_of_curves=5)
    plt.savefig(f'{save_fig_spectrum}/curves_tau_diff.{args.fig_format}')
    if args.verbose:
        plt.show()
    else:
        plt.close()

    _max = r_[networks_eta_avg[mask_max_list].argmax(axis=1)]
    a = (_max>.1) & (_max<.9)
    print('Number of maximus minor than .9: {:d}/{:d}'.format(a.sum(), len(r_)))

    ####################################################################################################################
    ####################################################################################################################

    # pos = nx.spring_layout(G, seed=1)  # nx.draw_spectral
    # nx.draw(G, with_labels=False, node_size=2, node_color='skyblue', font_size=10, font_weight='bold', pos=pos)
    # plt.show()
    # plt.close()

    # import matplotlib.pyplot as plt
    # import pydot
    # from networkx.drawing.nx_pydot import graphviz_layout
    # pos = graphviz_layout(G, prog="circo")
    # plt.figure(1, figsize=(60, 60))
    # nx.draw(G, pos, node_size=10)
    # plt.show(block=False)
    #
    # import igraph as ig
    # bigGraphAsTupleList = edges #(('a', 'b'), ('b', 'c'), ('b', 'd'), ..., ('c', 'e'))
    # g = ig.Graph.TupleList(bigGraphAsTupleList)
    # layout = g.layout("rt_circular")  # fr (fruchterman reingold), tree, circle, rt_circular (reingold_tilford_circular)
    # # bbox = size of picture
    # ig.plot(g, layout=layout, bbox=(10000, 10000), target='mygraph.png')
    # ig.show()