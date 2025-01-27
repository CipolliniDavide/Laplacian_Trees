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
from utils_spectral_entropy.thermo_efficiency import thermo_efficiency_by_key
from utils_spectral_entropy.make_plots import (plot_thermo_trajectory, plot_thermo_trajectory_separate,
                                               plot_eta_curves, plot_quantity_along_transition,
                                               plot_eta_dS_dF, plot_surface, plot_tau_vs_r, plot_von_neumann_ent)

# Generate the log-spaced array
log_min = -2  # Start value (10^-3)
log_max = 6  # Stop value (10^4)
num_points = 1000  # Number of points
tau_range = np.logspace(log_min, log_max, num=num_points)

import numpy as np
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import eigsh
import os


def plot_ds_vs_r(r_values, y_values, save_fold, save_name, x_fit_range = (-1e3, 1e13)):

    # Linear fit: d_s and r
    index_values_fit = np.where((r_values <= x_fit_range[1]) & (r_values >= x_fit_range[0]))[0]
    r_values_fit = r_values[index_values_fit]
    y_values_fit = y_values[index_values_fit]
    # Linear fit
    coefficients = np.polyfit(r_values_fit, y_values_fit, 1)  # Linear fit (degree 1)
    fitted_values = np.polyval(coefficients, r_values_fit)

    # Calculate R^2: R^2=0: model does not explain variance in the data, R^2=1 perfect
    residuals = y_values_fit - fitted_values
    ss_res = np.sum(residuals ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_values_fit - np.mean(y_values_fit)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)

    # Plotting
    fig, ax = plt.subplots()
    plt.plot(r_values, y_values, marker="o", label="Data")
    plt.plot(r_values_fit, fitted_values,
             "--",
             label=f"Fit: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}\n$R^2$ = {r_squared:.2f}")
    set_ticks_label(ax=ax, ax_label=r'$\mathbf{r}$', ax_type='x', data=r_values, ticks=r_values[::2])
    set_ticks_label(ax=ax, ax_label=r'$\mathbf{1/d_s}$', ax_type='y', data=y_values, ticks=np.hstack((y_values[::2], (y_values[-1]))))

    # plt.xlim(r_values[0], r_values[-1])
    # plt.ylim(r_values[0], r_values[-1])
    set_legend(ax=ax)
    plt.tight_layout()
    plt.savefig(save_fold+save_name, dpi=300)
    # plt.show()


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
    parser.add_argument('-s_dir', '--save_dir', default=f'{os.getcwd()}/OutputTest/', type=str, help='Name of dataset_dir.')
    parser.add_argument('-fig_form', '--fig_format', default='pdf', type=str, help='')
    parser.add_argument('-sp', '--spectrum_type', default='spectrum', type=str, help='')
    parser.add_argument("-v", "--verbose", action="store_true", help="Suppresses all output if False")
    parser.add_argument("-maxit", '--max_iter', default=50, type=int, help='Maximum number of iterations to include in the dataset')
    args = parser.parse_args()
    print(args)


    ########################################################################################################################
    ds_path_list = [
                    "./Dataset/boom-cifar10-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/",
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
        temp_fold = ds_path.replace('Dataset', 'OutputTest/{:s}/'.format(args.fig_format))
        df = read_csv_convert_to_dataframe(file_path=ds_path + 'dataset.csv',
                                           save_fold=f'{temp_fold}/',
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

        # plt.plot(r, trade_off, label='trade_off')
        # plt.axvline(r_max_theta, linestyle='--', label='max theta')
        # # Plot the shaded area for the intersection using axvspan
        # for i in range(len(intersection_x_values) - 1):
        #     axvspan_start = intersection_x_values[i]
        #     axvspan_end = intersection_x_values[i + 1]
        #     plt.axvspan(axvspan_start, axvspan_end, color='purple', alpha=0.06)
        # plt.title(f"{dataset_name}")
        # plt.show()

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


        ######################## Select r range #########################################
        # r_index_for_avg = np.where((r_ <= intersection_x_values.max()) & (r_ >= intersection_x_values.min()))[0]

        if "cifar10" in dataset_name:
            r_index_for_avg = np.where((r_ <= r_[-1]) & (r_ > .45))[0]
        r_index_for_avg = np.where((r_ <= r_[-1]) & (r_ > .45))[0]

        C = np.array([np.array(C_save[i]) for i in r_index_for_avg])

        index_tau_range_spectral_dim = np.where((tau_range < 20) & (tau_range >= 10))[0]

        C_plateau = C[:, index_tau_range_spectral_dim, :].mean()
        list_spectral_dimension_avg_over_r.append( 2*C_plateau)
        list_spectral_dimension_std_avg_over_r.append( (2*C[:, index_tau_range_spectral_dim, :]).std())

        gamma_temp_arr = np.zeros(len(r_index_for_avg))
        for ind, r_ind in enumerate(r_index_for_avg):
            power_law_save_fig = temp_fold + "/power_law/"
            os.makedirs(power_law_save_fig, exist_ok=True)
            try:
                gamma_temp_arr[ind] = plot_log_log_distribution(size_cc=[spectrum.mean(0)[r_ind]],
                                                  N=[r_[r_ind]],
                                                  x_lim_fit=(1e-3, .2),
                                                  title_leg='r',
                                                  save_dir=power_law_save_fig,
                                                  fig_name=f'powlaw_{dataset_name}_r={r_[r_ind]:.02f}',
                                                  fig_format=args.fig_format,
                                                  figsize=(5, 4),
                                                  cmap_='inferno',
                                                  y_lim=(1e-5, 10),
                                                  legend_flag=False,
                                                  make_fit_flag=True,
                                                  return_exponent=True,
                                                  # show=False
                                                  show=args.verbose
                                                  )
            except:
                gamma_temp_arr[ind] = np.nan
        print(f"Dataset {dataset_name}:")
        print(gamma_temp_arr)
        list_gamma_avg_over_r.append(gamma_temp_arr.mean(0))
        list_gamma_std_over_r.append(gamma_temp_arr.std(0))

        fig, ax = plt.subplots()
        ax.semilogx(tau_range, C.mean(0).mean(1))
        ax.semilogx(tau_range[index_tau_range_spectral_dim], C[:, index_tau_range_spectral_dim, :].mean(0).mean(1), c='red')
        ax.axhline(y=gamma_temp_arr.mean() + 1)
        ax.set_title(f"{dataset_name}")
        plt.show()
        plt.close()

        plot_ds_vs_r(r_values = r_[r_index_for_avg], y_values = 1 / (2 * (gamma_temp_arr + 1)),
                     save_fold=temp_fold.rsplit("/boom")[0],
                     save_name=f"{dataset_name}_spectral_dim_vs_r.{args.fig_format}",
                     x_fit_range=(-1e13, 1e13) if "boom-mnist" not in dataset_name else (.7, .85)
                     )

        ###############################################################################################################
        # Efficiency
        # Compute ensemble efficiency, dS and dF
        # networks_eta_avg, networks_eta_std, \
        #     dS_avg, dS_std, \
        #     dF_avg, dF_std = thermo_efficiency_by_key(tau_range=tau_range, spectrum=spectrum)
        #
        # #### C'e' un plateu: e' relazionato con C_0 e d_s?
        # for r_ind, gamma in zip(r_index_for_avg[::2], gamma_temp_arr[::2]):
        #     plt.semilogx(tau_range, networks_eta_avg.T[r_ind], label=f'r={r_[r_ind]:.02f}')
        #     plt.axhline(y=np.log(gamma + 1))
        # plt.xlim(1e-1, 1e6)
        # plt.legend()
        # plt.show()
        #
        #
        # tau_lim = (10, 1e2)
        # tau_mask = np.where((tau_range >= tau_lim[0]) & (tau_range <= tau_lim[1]))[0][::20]
        #
        # for ind_tau in tau_mask:
        #     plt.plot(1 / (2 * (gamma_temp_arr + 1)), networks_eta_avg[ind_tau, r_index_for_avg],
        #              marker="o",
        #              label=r'$\tau=$' + f"{tau_range[ind_tau]:.1e}")
        # plt.xlabel(r'$d_s$')
        # plt.ylabel(r'$\eta$')
        # plt.legend()
        # plt.show()
        #
        # plot_thermo_trajectory_separate(tau_range=tau_range,
        #                                 r=r_,
        #                                 # vmax=-1.3,
        #                                 networks_eta=networks_eta_avg,
        #                                 fig_name=r'$\mathbf{\eta}$',
        #                                 add_yticks_heat=[np.log10(10), np.log10(100)],
        #                                 y_label=r'$\mathbf{\eta}$',
        #                                 # save_dir=save_fig_spectrum,
        #                                 figsize=(8, 5),
        #                                 title='',
        #                                 num_xticks=len(r_) // 4,
        #                                 x_ticks=[.1, .4, .6, .8],
        #                                 tau_lim=(10, 12),
        #                                 number_of_curves=10,
        #                                 heat_num_yticks=2,
        #                                 valfmt_y="{x:.0f}",
        #                                 fig_format=args.fig_format,
        #                                 # show=args.verbose
        #                                 show=True
        #                                 )

        ##############################################################################################################



    a=0