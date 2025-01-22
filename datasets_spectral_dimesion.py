import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import RawTextHelpFormatter

from helpers.visual_utils import get_set_larger_ticks_and_labels, set_ticks_label, set_legend
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import pandas as pd

from dataset_complexity_estimation import train_mlp, load_dataset_file, compute_intrinsic_dimensions, remove_repeated_samples
from helpers_dataset.helpers_dataset import read_csv_convert_to_dataframe, check_dataset, load_key_by_
from utils_spectral_entropy.get_tau_max_specific_heat import get_tau_max_specific_heat
from utils_spectrum.log_log_distribution import plot_log_log_distribution

# Generate the log-spaced array
log_min = -2  # Start value (10^-3)
log_max = 6  # Stop value (10^4)
num_points = 1000  # Number of points
tau_range = np.logspace(log_min, log_max, num=num_points)

import numpy as np
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import eigsh


def shadow_intersct_area(r, trade_off, trade_off_error):
    from scipy.interpolate import interp1d

    # Interpolate the trade_off array for denser points
    f_trade_off = interp1d(r, trade_off, kind='linear')
    r_dense = np.linspace(min(r), max(r), 1000)
    trade_off_dense = f_trade_off(r_dense)

    # Find the indices where the intersection occurs
    intersection_indices = np.where((trade_off_dense >= trade_off.max() - trade_off_error[np.argmax(trade_off)]) &
                                    (trade_off_dense <= trade_off.max() + trade_off_error[np.argmax(trade_off)]))[0]

    # Find the corresponding x values at the intersection
    intersection_x_values = r_dense[intersection_indices]

    # Plot the trade_off curve
    # ax.plot(r_dense, trade_off_dense, color='blue', label='Trade-off')

    # Plot the shaded area for the intersection using axvspan
    # for i in range(len(intersection_x_values) - 1):
    #     axvspan_start = intersection_x_values[i]
    #     axvspan_end = intersection_x_values[i + 1]
    #     ax.axvspan(axvspan_start, axvspan_end, color='purple', alpha=0.06)
    return intersection_x_values


if  __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    # parser.add_argument('-s_dir', '--save_dir', default=f'{os.getcwd()}/Output/PortegysTree', type=str, help='Name of dataset_dir.')
    parser.add_argument('-fig_form', '--fig_format', default='pdf', type=str, help='')
    parser.add_argument('-sp', '--spectrum_type', default='spectrum', type=str, help='')
    parser.add_argument("-v", "--verbose", action="store_true", help="Suppresses all output if False")
    parser.add_argument("-maxit", '--max_iter', default=50, type=int, help='Maximum number of iterations to include in the dataset')
    args = parser.parse_args()
    print(args)

    import networkx as nx

    # N = 10
    # G = nx.grid_graph(dim=(N, N, N), periodic=True)
    # nx.draw_networkx(G, with_labels=False, node_size=.1)
    # plt.show()
    #
    # gamma = plot_log_log_distribution(size_cc=[cube_spectrum],
    #                                   N=[N],
    #                                   x_lim_fit=(1e-3, 1 / 10),
    #                                   title_leg='r',
    #                                   # save_dir=save_fold,
    #                                   fig_name=f'powlaw_cubeSpectrum_N={N:d}',
    #                                   fig_format=args.fig_format,
    #                                   figsize=(5, 4),
    #                                   cmap_='inferno',
    #                                   y_lim=(1e-5, 10),
    #                                   legend_flag=False,
    #                                   # make_fit_flag=True,
    #                                   return_exponent=True,
    #                                   show=True
    #                                   # show=args.verbose
    #                                   )

    ########################################################################################################################
    ds_path_list = ["./Dataset/boom-cifar10-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/",
                    "./Dataset/boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=3566897019/",
                    "./Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=64186134/",
                    "./Dataset/boom-nist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/"
                    ][::-1]

    # ds_path_list = ["./Dataset/boom-cifar10-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/",
    #                 "./Dataset/boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1/",
    #                 "./Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1/",
    #                 "./Dataset/boom-nist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/"
    #                 ]

    list_spectral_dimension = []
    list_spectral_dimension_std = []
    gamma_list = []

    list_spectral_dimension_avg_over_r = []
    list_spectral_dimension_std_avg_over_r = []
    list_gamma_avg_over_r = []
    list_gamma_std_over_r = []

    for ds_path in ds_path_list:
        # Load dataset containing algorithm performance
        dataset_name = ds_path.split('boom-')[1].split('-')[0]
        temp = ds_path.replace('Dataset', 'OutputTest/{:s}/'.format(args.fig_format))
        df = read_csv_convert_to_dataframe(file_path=ds_path + 'dataset.csv',
                                           save_fold=f'{temp}/',
                                             figformat=args.fig_format,
                                             keep_r_=(0, 1))
        r_range=(0, .85)
        if (r_range[0] is not None) and (r_range[1] is not None):
            df = df[(df['R'] >= r_range[0]) & (df['R'] <= r_range[1])]

        r = df['R'].unique()
        # if x_ticks is None:
        #     x_ticks = r[::x_tick_every]

        grouped_avg = df.groupby('R').mean()
        grouped_error = df.groupby('R').std()  # / np.sqrt(df.groupby('R').count())

        trade_off = grouped_avg['trade_off'].values
        trade_off_error = grouped_error['trade_off'].values
        intersection_x_values = shadow_intersct_area(r=r, trade_off=trade_off, trade_off_error=trade_off_error)

        r_max_theta = r[np.argmax(trade_off)]

        plt.plot(r, trade_off, label='trade_off')
        plt.axvline(r_max_theta, linestyle='--', label='max theta')
        # Plot the shaded area for the intersection using axvspan
        for i in range(len(intersection_x_values) - 1):
            axvspan_start = intersection_x_values[i]
            axvspan_end = intersection_x_values[i + 1]
            plt.axvspan(axvspan_start, axvspan_end, color='purple', alpha=0.06)
        plt.title(f"{dataset_name}")
        plt.show()

        # Load and prepare dataset according to limit on R and independent iterations
        load_path_ds_nw = ds_path.replace('Dataset', 'OutputTest/{:s}/'.format(args.fig_format))
        df_nw = pd.read_hdf(f'{load_path_ds_nw}/dataset_nw.h5', key='data').sort_values(by=['R', 'iter'])
        df_nw = df_nw[(df_nw['R'] < .9) & (df_nw['iter'] < args.max_iter)]
        # df_nw = df_nw[(df_nw['R'] < 2) & (df_nw['iter'] < args.max_iter)]

        r_ = np.unique(df_nw['R'].values)
        # Check that R is equally represented
        check_dataset(df=df_nw, max_iter=args.max_iter)

        save_prop_dir = ds_path+'prop/'
        spectrum = load_key_by_(max_iter=args.max_iter, key_name=args.spectrum_type, load_dir=save_prop_dir,
                                r_range=(r_.min(), r_.max()))
        (mask_of_peaks_list, tau_firstpeak, tau_lastpeak, mask_max_list, tau_maxpeak,
         critical_entropy_firstpeak, critical_entropy_lastpeak, critical_entropy_maxpeak,
         spec_heat_firstpeak, spec_heat_lastpeak, spec_heat_maxpeak, entropy_save, C_save) = get_tau_max_specific_heat(
            spectrum=spectrum,
            r_list=r_,
            tau_range=tau_range,
            )

        ############################ One sigle value of r= argmax(\theta) ##############################################
        # index_argmax_theta = np.argwhere(r_ == r_max_theta)[0][0]
        # C = C_save[index_argmax_theta]
        # index_tau_range_spectral_dim = np.where((tau_range < 12) & (tau_range >= 9.5))[0]
        #
        # C_plateau = C.mean(1)[index_tau_range_spectral_dim]
        #
        # list_spectral_dimension.append(2*C_plateau.mean())
        # list_spectral_dimension_std.append((2 * C_plateau).std())
        #
        # avg_spec = spectrum.mean(0)[index_argmax_theta]
        #
        # gamma =plot_log_log_distribution(size_cc=[avg_spec],
        #                           N=[r_max_theta],
        #                           x_lim_fit=(1e-3, .2),
        #                           title_leg='r',
        #                           # save_dir=save_fold,
        #                           fig_name=f'powlaw_{dataset_name}_r={r_max_theta}',
        #                           fig_format=args.fig_format,
        #                           figsize=(5, 4),
        #                           cmap_='inferno',
        #                           y_lim=(1e-5, 10),
        #                           legend_flag=False,
        #                           make_fit_flag=True,
        #                           return_exponent=True,
        #                           show=True
        #                           # show=args.verbose
        #                           )
        # gamma_list.append(gamma)

        ######################## average over r purple interval #########################################
        r_index_for_avg = np.where((r_ < intersection_x_values.max()) & (r_ > intersection_x_values.min()))[0]
        if "cifar10" in dataset_name:
            # We remove r=0.35 as gives a gamma value that is visibly an outlier, moreover the
            # trade-off theta is much less teh the remaining r values
            r_index_for_avg = r_index_for_avg[1:]

        C = np.array([np.array(C_save[i]) for i in r_index_for_avg])

        index_tau_range_spectral_dim = np.where((tau_range < 20) & (tau_range >= 10))[0]

        C_plateau = C[:, index_tau_range_spectral_dim, :].mean()
        list_spectral_dimension_avg_over_r.append( 2*C_plateau)
        list_spectral_dimension_std_avg_over_r.append( (2*C[:, index_tau_range_spectral_dim, :]).std())

        gamma_temp_arr = np.zeros(len(r_index_for_avg))
        for ind, r_ind in enumerate(r_index_for_avg):
            gamma_temp_arr[ind] = plot_log_log_distribution(size_cc=[spectrum.mean(0)[r_ind]],
                                              N=[r_[r_ind]],
                                              x_lim_fit=(1e-3, .2),
                                              title_leg='r',
                                              # save_dir=save_fold,
                                              fig_name=f'powlaw_{dataset_name}_r={r_max_theta}',
                                              fig_format=args.fig_format,
                                              figsize=(5, 4),
                                              cmap_='inferno',
                                              y_lim=(1e-5, 10),
                                              legend_flag=False,
                                              make_fit_flag=True,
                                              return_exponent=True,
                                              show=True
                                              # show=args.verbose
                                              )
        print(f"Dataset {dataset_name}:")
        print(gamma_temp_arr)
        list_gamma_avg_over_r.append(gamma_temp_arr.mean(0))
        list_gamma_std_over_r.append(gamma_temp_arr.std(0))


        plt.semilogx(tau_range, C.mean(0).mean(1))
        plt.semilogx(tau_range[index_tau_range_spectral_dim], C[:, index_tau_range_spectral_dim, :].mean(0).mean(1), c='red')
        plt.axhline(y=gamma_temp_arr.mean() + 1)
        plt.title(f"{dataset_name}")
        plt.show()

    colors = sns.color_palette(n_colors=3)[::-1]
    dataset_names = ['NIST', 'MNIST', 'FashionMNIST', 'CIFAR10']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), tight_layout=True)

    # plt.errorbar(np.arange(len(ds_path_list)), y=list_spectral_dimension, yerr=list_spectral_dimension_std,  label=r'$2C_0$')
    # plt.scatter(np.arange(len(ds_path_list)), 2*(np.array(gamma_list)+1), c='red', label=r'$2\cdot(\gamma+1)$')

    ax.errorbar(
        np.arange(len(ds_path_list)),
        y=list_spectral_dimension_avg_over_r,
        yerr=list_spectral_dimension_std_avg_over_r,
        fmt="^",
        label=r'$2C_0$',
        c="blue",
        capsize=6,  # Adds "caps" at the end of error bars
        elinewidth=2,  # Increases the width of error bars
        alpha=0.5  # Makes error bars semi-transparent
    )

    ax.errorbar(
        np.arange(len(ds_path_list)),
        y=2 * (np.array(list_gamma_avg_over_r) + 1),
        yerr=list_gamma_std_over_r,
        fmt="o",
        label=r'$2\cdot(\gamma+1)$',
        c="red",
        capsize=6,
        elinewidth=2,
        alpha=0.8
    )
    ax.set_ylabel(r"$\mathbf{d_s}$")
    # Set tick positions and labels
    ax.set_xticks(range(len(dataset_names)))
    ax.set_xticklabels(dataset_names, rotation=45)
    # ax[1].set_ylabel('Lempel-Ziv Complexity')
    # ax[1].set_yscale('log')
    get_set_larger_ticks_and_labels(ax=ax)
    set_legend(ax=ax, title='')
    number_of_nodes = ds_path.rsplit('f=')[1].split(",")[0]
    plt.savefig(f"OutputTest/{args.fig_format}/spectral_dimension_datasets_{number_of_nodes}.{args.fig_format}", dpi=300)
    plt.show()

