import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import os
import glob
import re
from tqdm import tqdm
from itertools import product
from scipy.linalg import eigh

from .convert_tree_to_nx import convert_tree_to_nx
from helpers.utils import ensure_dir, pickle_save, pickle_load
from helpers.visual_utils import get_set_larger_ticks_and_labels, set_ticks_label
from helpers.spanning_trees import number_of_spanning_trees

def filter_file_list(file_list: list, max_value: int):
    # Filter the file list
    filtered_files = []
    for file_path in file_list:
        match = re.search(r'iter=(\d+)', file_path)
        if match:
            iter_number = int(match.group(1))
            if iter_number < max_value:
                filtered_files.append(file_path)
    return filtered_files


def check_dataset(df: pd.DataFrame, max_iter: int):
    value_counts = df['R'].value_counts()
    if value_counts.max() == value_counts.min():
        print("\nChosen R values are equally represented.")
    else:
        keys_with_less_values = value_counts[value_counts < max_iter].index.tolist()
        for key in keys_with_less_values:
            available_iter = df[df['R'] == key]['iter'].values
            set_a1 = set(available_iter)
            set_a2 = set(np.arange(max_iter))
            # Find missing values in a1 compared to a2
            missing_values = list(set_a2 - set_a1)
            numbers_string = ','.join(map(str, missing_values))
            print('Missing simulations:\n\tr={}: iters={:s} '.format(key, numbers_string))
        raise Exception("\nR values are not equally represented in terms of independent iterations.\nMissing i")


def load_key_by_(key_name, load_dir: str, max_iter: int, r_range: tuple=(0, .9)):
    """
    Load key by r
    :param key_name:
    :param load_dir:
    :return: output array of shape (batch, number of r values, size of the key)
    """
    file_list = sorted(glob.glob('{:s}/*'.format(load_dir)))
    file_list = filter_file_list(file_list, max_iter)

    dict_temp = order_files_by_r(file_list=file_list)
    dict_files = {key: value for key, value in dict_temp.items() if r_range[0] <= key <= r_range[1]}

    # r_ = np.fromiter(dict_files.keys(), dtype=np.float32)
    temp = pickle_load(filename=file_list[0])[key_name]

    if hasattr(temp, '__len__'):
        dtype = temp.dtype
        shape_temp = temp.shape
    else:
        dtype = float
        shape_temp = (1,)

    first_key = next(iter(dict_files))
    output_arr = np.empty(shape=(len(dict_files[first_key]), len(dict_files),) + shape_temp, dtype=dtype)

    for i, key in tqdm(enumerate(dict_files.keys()), total=len(dict_files)):
        for b, file_name in enumerate(dict_files[key]):
            try:
                ind_iter = int(file_name.rsplit('iter=')[1].rsplit('.pkl')[0])
                output_arr[ind_iter, i] = pickle_load(filename=file_name)[key_name]
            except:
                print('Problem', key)
                print(file_name)

    return output_arr


def order_files_by_r(file_list):
    # Initialize a dictionary to store file names for each 'r' value
    file_dict = {}
    # Iterate through the file names
    for file_name in file_list:
        # Extract the value of 'r' from the file name
        r_value = np.round(float(file_name.split('-r=')[1].split('-')[0]), 2)
        # Check if the 'r' value already exists in the dictionary
        if r_value in file_dict:
            file_dict[r_value].append(file_name)
        else:
            file_dict[r_value] = [file_name]
    return file_dict


def dict_of_properties(G: nx.Graph, key):
    # ls = np.sort(nx.laplacian_spectrum(G))
    # ls_rw = np.sort(nx.normalized_laplacian_spectrum(G))

    # eigenvalues, eigvec = eigh(nx.laplacian_matrix(G).todense(), eigvals_only=False)
    # eigenvalues_rw, eigvec_rw = eigh(nx.normalized_laplacian_matrix(G).todense(), eigvals_only=False)

    eigenvalues = eigh(nx.laplacian_matrix(G).todense(), eigvals_only=True)
    eigenvalues_rw = eigh(nx.normalized_laplacian_matrix(G).todense(), eigvals_only=True)

    clustering = list(nx.clustering(G).values())
    return {'R': key,
            # 'iter': iter,
            'connected_components': nx.number_connected_components(G),
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'avg_clustering': np.mean([cc for cc in clustering if cc > 0]),
            'clustering': clustering,
            'number_of_spanning_trees': number_of_spanning_trees(nx.laplacian_matrix(G).toarray()),
            'number_of_spanning_trees_Lrw': number_of_spanning_trees(nx.normalized_laplacian_matrix(G).toarray()),
            'diameter': nx.diameter(G),
            'degrees': np.array([d for n, d in G.degree()]),
            'spectrum': eigenvalues,
            'spectrum_rw': eigenvalues_rw,
            # 'eigvec': eigvec,
            # 'eigvec_rw': eigvec_rw
            }


def create_dataset_Portegys(save_fold_prop: str, save_fold_ds: str, dict_files: dict = None, crt_properties: bool = True):
    if crt_properties:
        ensure_dir(save_fold_prop)
        progress_bar = tqdm(total=len(dict_files))
        for _, key in enumerate(dict_files.keys()):
            for _, file_name in enumerate(dict_files[key]):
                G = convert_tree_to_nx(file_name)
                try:
                    pickle_save(filename=save_fold_prop + file_name.rsplit('/', 1)[-1].replace('.tree', '.pkl'),
                                obj=dict_of_properties(G, key=key))
                except:
                    print('\nError converting tree to nx:\n\t{:s}'.format(file_name))
            progress_bar.update(1)

    file_list = glob.glob(save_fold_prop + '*')
    samples = []
    for f in file_list:
        iter = int(f.split('-iter=')[1].split('.pkl')[0])
        sam = pickle_load(filename=f)
        sam['iter'] = iter
        # if (sam['n_nodes'] != 2000 or sam['n_edges'] != 2000 or
        # if sam['number_of_spanning_trees'] < .5:
        if np.max(sam['spectrum']) > 1e6:
            print(f)
        try:
            samples.append(sam)
        except:
            print(f)
    df = pd.DataFrame(samples)
    # Save DataFrame to HDF5
    ensure_dir(save_fold_ds)
    df.to_hdf(f'{save_fold_ds}/dataset_nw.h5', key='data', mode='w')
    # df.to_csv(f'{save_fold_ds}/dataset_nw.csv')


def gaussian_clouds(num_clouds=4, num_samples=2000, x_range=(0, 2 * np.pi), num_std=3):
    cloud_size = num_samples // num_clouds

    std_dev_temp = (x_range[1] - x_range[0]) / (num_clouds ** 2)
    mean = np.linspace(x_range[0] + num_std * std_dev_temp,
                       x_range[1] - num_std * std_dev_temp,
                       num=num_clouds,
                       endpoint=True)
    std_dev = (mean.max() - mean.min()) / (num_clouds ** 2)

    # mean = np.random.uniform(low=x_range[0], high=x_range[1])  # Random mean
    # std_dev = np.random.uniform(low=1, high=.01)  # Random standard deviation

    # Generate clouds of points using Gaussian distributions
    clouds = []
    for c in range(num_clouds):
        cloud_points = np.random.normal(loc=mean[c], scale=std_dev, size=cloud_size)
        clouds.append(cloud_points)

    # Flatten the list of clouds to get all points in a single list
    all_points = np.concatenate(clouds)
    np.random.shuffle(all_points)

    # Plot the clouds of points
    # plt.figure(figsize=(8, 4))
    # plt.hist(all_points, bins=30, alpha=0.5, density=True)
    # plt.title("Clouds of Points in 1D using Gaussian Distributions")
    # plt.xlabel("x")
    # plt.ylabel("Density")
    # plt.xlim(x_range)
    # # plt.savefig(save_samples_dir.rsplit('/', 2)[0] + '/samples_distribution.png')
    # plt.show()

    return all_points


def read_csv_convert_to_dataframe(file_path: str, save_fold: str = os.getcwd(), figformat: str = 'png',
                                  keep_r_=None, show=False):
    # number_of_samples = 2000
    # N_dist_max = number_of_samples*(number_of_samples-1)
    # Read the CSV file into a Pandas DataFrame
    df_temp = pd.read_csv(file_path, header=None)

    df = pd.DataFrame()

    # Extract the relevant columns
    # df['unit'] = df_temp[8].values.astype(str)
    df['R'] = df_temp[1].values.astype(float)
    df['Accu'] = df_temp[3].values.astype(float)
    df['Ndist'] = df_temp[5].astype(int)  #/ N_dist_max
    df['tpu'] = df_temp[7].astype(float)
    # df['Accu_norm'] = - np.log(df_temp[3].values.astype(float))

    df['log(Ndist)'] = np.log(df['Ndist'])
    df['trade_off'] = df['Accu'] / df['log(Ndist)']

    if keep_r_ is not None:
        df = df[(df['R'] >= keep_r_[0]) & (df['R'] <= keep_r_[1])]

    return df


def plot_algorithmic_accuracy(df: pd.DataFrame, r_range: tuple=(None, None),
                              fig_name: str='alg_efficiency',
                              fig_format: str = 'png', save_dir: str = os.getcwd(),
                              show: bool = False, fig_size: tuple=(16, 4), x_tick_every: int=2,
                              valfmt_x="{x:.2f}",
                              vbar_max_eff: bool=True,
                              vbar_max_acc: bool=False,
                              x_ticks=None,
                              ylim_theta=None,
                              ylim_acc =None,
                              fontsize_ticks=20,
                              fontsize_labels=35,
                              fontsize_ticks_multiplot=20,
                              fontsize_labels_multiplot=35,
                              fig_size_trade_off=(8.5, 5.5),
                              fontsize_legend_title=30,
                              ):

    if (r_range[0] is not None) and (r_range[1] is not None):
        df = df[(df['R'] >= r_range[0]) & (df['R'] <= r_range[1])]

    r = df['R'].unique()
    if x_ticks is None:
        x_ticks = r[::x_tick_every]

    grouped_avg = df.groupby('R').mean()
    grouped_error = df.groupby('R').std()  # / np.sqrt(df.groupby('R').count())

    trade_off = grouped_avg['trade_off'].values
    trade_off_error = grouped_error['trade_off'].values

    # Create subplots
    # keys_to_plot = ['Accu', 'Ndist', 'log(Ndist)']
    # labels = ['Accuracy', r'$N_{dist}$', r'$\log(N_{dist})$']  # 'Ndist_norm']
    # fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10), sharex=True, layout='tight')
    # axs = axs.flat

    keys_to_plot = ['Accu', 'log(Ndist)']
    labels = ['Accuracy', r'$\mathbf{\log(N_{dist})}$']  # 'Ndist_norm']
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=fig_size, sharex=True, layout='tight')
    axs = axs.flat

    fontsize=14
    for i, (key, lab) in enumerate(zip(keys_to_plot, labels)):
        ax = axs[i]
        if key == 'Ndist':
            ax.semilogy(r, grouped_avg[key], label=key, linewidth=3, marker='o', color='blue')
        else:
            ax.plot(r, grouped_avg[key], label=key, linewidth=3, marker='o', color='blue')
        ax.fill_between(r, grouped_avg[key] - grouped_error[key], grouped_avg[key] + grouped_error[key],
                        color='gray', alpha=0.5)  # Add standard error

        if key == 'Accu':
            ax.set_ylim(ylim_acc)
        # get_set_larger_ticks_and_labels(ax=ax)
        # ax.set_ylabel(lab,  fontsize=fontsize)
        set_ticks_label(ax=ax,
                        ax_label=lab,
                        valfmt=valfmt_x, ax_type='y', data=grouped_avg[key],
                        num=2,
                        fontdict_ticks_label={'weight': 'bold', 'size': fontsize_ticks_multiplot},
                        fontdict_label={'weight': 'bold', 'size': fontsize_labels_multiplot, 'color': 'black'},
                        )

        # ax.set_xlabel(r'$\mathbf{r}$', fontsize=fontsize)
        set_ticks_label(ax=ax,
                        ax_label=r'$\mathbf{r}$',
                        ticks=x_ticks + r[np.argmax(trade_off)], valfmt=valfmt_x, ax_type='x', data=r,
                        fontdict_ticks_label={'weight': 'bold', 'size': fontsize_ticks_multiplot},
                        fontdict_label={'weight': 'bold', 'size': fontsize_labels_multiplot, 'color': 'black'},
                        )
        # ax.grid(True, which='both', axis='x')
        # set_legend(ax=ax, title=f'r={legend_title}')
        if vbar_max_eff:
            ax.axvline(x=r[np.argmax(trade_off)], linestyle='--', linewidth=3, color='red')
        if vbar_max_acc:
            max_value = np.max(grouped_avg['Accu'])
            # Find all indices of the maximum value
            max_indices = np.where(grouped_avg['Accu'] == max_value)[0]
            for max_ind in max_indices:
                ax.axvline(x=r[max_ind], linestyle='--', linewidth=3, color='green')

    ################################ Trade-off #########################################3

    ax = axs[-1]
    ax.plot(r, trade_off, linewidth=3, marker='o', color='blue')
    ax.fill_between(r, trade_off - trade_off_error, trade_off + trade_off_error,
                    color='gray', alpha=0.5)  # Add standard error
    ax.axvline(x=r[np.argmax(trade_off)], linestyle='--', linewidth=3, color='red')
    # ax.axvline(x=r[np.argmax(trade_off+trade_off_error)], linestyle='--', linewidth=3, color='red')
    condition = ((trade_off >= trade_off.max() - trade_off_error[np.argmax(trade_off)]) &
                 (trade_off <= trade_off.max() + trade_off_error[np.argmax(trade_off)]))
    ax.axhspan(ymin=trade_off.max() - trade_off_error[np.argmax(trade_off)],
               ymax=trade_off.max() + trade_off_error[np.argmax(trade_off)], color='yellow', alpha=.2)

    def shadow_intersct_area(ax, r, trade_off, trade_off_error):
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
        for i in range(len(intersection_x_values) - 1):
            axvspan_start = intersection_x_values[i]
            axvspan_end = intersection_x_values[i + 1]
            ax.axvspan(axvspan_start, axvspan_end, color='purple', alpha=0.06)

    shadow_intersct_area(ax=ax, r=r, trade_off=trade_off, trade_off_error=trade_off_error)
    # ax.plot(r, trade_off_tpu, linewidth=3, marker='o', color='purple')
    # ax.set_ylabel(r'$\mathbf{\theta}$', fontsize=fontsize)
    # get_set_larger_ticks_and_labels(ax=ax)

    set_ticks_label(ax=ax,
                    ax_label="r",
                    ticks=x_ticks+[r[np.argmax(trade_off)]],
                    valfmt=valfmt_x,
                    ax_type='x',
                    data=r,
                    fontdict_ticks_label={'weight': 'bold', 'size': fontsize_ticks_multiplot},
                    fontdict_label={'weight': 'bold', 'size': fontsize_labels_multiplot, 'color': 'black'},
                    )
    set_ticks_label(ax=ax,
                    ax_label=r'$\mathbf{\theta}$',
                    valfmt=valfmt_x,
                    ax_type='y',
                    ticks=[ylim_theta[0], ylim_theta[1]] if ylim_theta is not None else None,
                    data=trade_off,
                    num=2,
                    fontdict_ticks_label={'weight': 'bold', 'size': fontsize_ticks_multiplot},
                    fontdict_label={'weight': 'bold', 'size': fontsize_labels_multiplot, 'color': 'black'},
                    )

    # ax.grid(True, which='both', axis='x')
    ax.set_xlim(left=0)
    if ylim_theta is not None:
        ax.set_ylim(ylim_theta)
    plt.savefig('{:s}/{:s}.{:s}'.format(save_dir, fig_name, fig_format), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    ################################################################################
    # Save only the trade-off plot
    fig_trade_off, ax_trade_off = plt.subplots(figsize=fig_size_trade_off)
    ax_trade_off.plot(r, trade_off, linewidth=3, marker='o', color='blue')
    ax_trade_off.fill_between(r, trade_off - trade_off_error, trade_off + trade_off_error, color='gray', alpha=0.5)
    ax_trade_off.axvline(x=r[np.argmax(trade_off)], linestyle='--', linewidth=3, color='red')
    # ax_trade_off.axvline(x=r[np.argmax(trade_off + trade_off_error)], linestyle='--', linewidth=3, color='red')

    # shadow_intersct_area(ax=ax_trade_off, r=r, trade_off=trade_off, trade_off_error=trade_off_error)
    # ax_trade_off.axhspan(ymin=trade_off.max() - trade_off_error[np.argmax(trade_off)],
    #            ymax=trade_off.max() + trade_off_error[np.argmax(trade_off)], color='yellow', alpha=.2)

    set_ticks_label(ax=ax_trade_off,
                    # ax_label="r",
                    ax_label='',
                    ticks=x_ticks, # + [r[np.argmax(trade_off)]],
                    valfmt=valfmt_x,
                    ax_type='x',
                    data=r,
                    fontdict_ticks_label={'weight': 'bold', 'size': fontsize_ticks},
                    fontdict_label={'weight': 'bold', 'size': fontsize_labels, 'color': 'black'},
                    )
    set_ticks_label(ax=ax_trade_off,
                    # ax_label=r'$\mathbf{\theta}$',
                    ax_label='',
                    valfmt=valfmt_x,
                    ax_type='y',
                    ticks=[ylim_theta[0], ylim_theta[1]] if ylim_theta is not None else None,
                    data=trade_off,
                    num=2,
                    fontdict_ticks_label={'weight': 'bold', 'size': fontsize_ticks},
                    fontdict_label={'weight': 'bold', 'size': fontsize_labels, 'color': 'black'},
                    )
    # # Move labels inside the heat map
    ax_trade_off.text(0.15, 0.1, r'$\mathbf{r}$', transform=ax_trade_off.transAxes,
             color='black', fontsize=fontsize_labels, fontweight='bold', va='center', ha='center')
    #
    ax_trade_off.text(-0.09, .55, r'$\mathbf{\theta}$', transform=ax_trade_off.transAxes,
             color='black', fontsize=fontsize_labels, fontweight='bold', va='center', ha='center', rotation=90)
    ax_trade_off.tick_params(axis='both', which='both', width=3, length=7)
    ax_trade_off.set_xlim((r.min()-0.015, r.max() + .015))

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{fig_name}_trade_off.{fig_format}', dpi=300)
    plt.close(fig_trade_off)


def plot_ntw_properties(r_: np.array, df: pd.DataFrame, save_dir: str = './', show: bool = False,
                        fig_format: str = 'png'):
    for key, ylabel in zip(['connected_components',
                            'diameter',
                            'number_of_spanning_trees',  # 'number_of_spanning_trees_Lrw',
                            # 'avg_clustering'
                            ],
                           [r'$\mathbf{CC}$',
                            r'$\mathbf{\bar{d}/N}$',
                            't(G)',
                            # 'Avg clustering'
                            ]):
        y = df.groupby('R')[key].mean()
        yerr = df.groupby('R')[key].std()
        # arr = load_key_by_(key_name=key, load_dir=save_prop_dir)[..., 0]
        # y = arr.mean(axis=0)
        # yerr = arr.std(axis=0)

        figsize = (9, 5)
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)  # Change layout to tight_layout
        if 'diameter' in key:
            std = yerr
            # ax.error(r_, y=y, yerr=std, marker='o', c='black')
            ax.plot(r_, y/np.unique(df['n_nodes'])[0], marker='o', c='black')
            ax.set_xlabel(r'$\mathbf{r}$')
            ax2 = ax.twinx()
            ax2.plot(r_, std, color='blue')
            ax2.set_ylabel(r'$\mathbf{Var(\bar{d})}$', c='blue')
            ax2.tick_params(axis='y', colors='blue')
            ax2.yaxis.label.set_color('blue')
            get_set_larger_ticks_and_labels(ax=ax2)

        else:
            ax.plot(r_, y, marker='o', c='black')
            ax.set_xlabel(r'$\mathbf{r}$')
        ax.set_ylabel(ylabel)
        get_set_larger_ticks_and_labels(ax=ax, num_ticks_x=len(r_) // 4)
        # plt.title('Mean Diameter as a Function of alpha')
        if 'tree' in key:
            ax.set_ylim(bottom=y.min() - 1, top=y.max() + 1)
        plt.savefig(f'{save_dir}/{key}.{fig_format}')
        if show:
            plt.show()
        else:
            plt.close()


###############################################################################
# from auxiliary_scripts.PopularitySimilarity.network_optimization_model import NetworkOptimizationModel
# def create_samples_optimization_model(n=2000, m=1, iter_start=1, iter_end=15,
#                                       compute_dist='popularity_fading',
#                                       alpha_range=np.arange(.1, 11, .1),
#                                       distribution='uniform',
#                                       save_fold=os.getcwd()
#                                       ):
#     save_samples_dir = save_fold #+ '/samples/'
#     ensure_dir(save_samples_dir)
#
#     x_range = (0, 2 * np.pi)
#
#     print('\nCreating samples...\n')
#     total_iterations = len(range(iter_start, iter_end + 1)) * len(alpha_range)
#     progress_bar = tqdm(total=total_iterations)
#
#     for iter, alpha in tqdm(product(range(iter_start, iter_end + 1), alpha_range)):
#
#         if distribution == 'uniform':
#             theta_t_list = [np.random.uniform(x_range[0], x_range[1]) for i in range(n)]
#         elif distribution == 'gaussian_clouds':
#             theta_t_list = gaussian_clouds(num_clouds=5, x_range=x_range, num_samples=n, num_std=3)
#             theta_t_list = np.clip(theta_t_list, x_range[0], x_range[1])
#         else:
#             raise ValueError("Distribution choice needs to be either gaussian_clouds or uniform.")
#
#         # Create a Network Optimization Model with parameter m
#         model = NetworkOptimizationModel(m=m, alpha=alpha, compute_dist=compute_dist)
#
#         # Add nodes iteratively
#         for t in range(1, n + 1):
#             model.add_node(t, theta_t=theta_t_list[t - 1])
#
#         G = model.get_graph()
#         nx.write_graphml(G, "{:s}/graph-r={:06.2f}-iter={:05d}.graphml".format(save_samples_dir,
#                                                                                alpha, iter))
#         # Update progress bar
#         progress_bar.update(1)
#     # Close the progress bar
#     progress_bar.close()
#
#     # Plot the clouds of points
#     plt.figure(figsize=(8, 4))
#     plt.hist(theta_t_list, bins=30, alpha=0.5, density=True)
#     plt.title("Clouds of Points in 1D using Gaussian Distributions")
#     plt.xlabel("x")
#     plt.ylabel("Density")
#     plt.xlim(x_range)
#     plt.savefig(save_samples_dir.rsplit('/', 2)[0] + '/samples_distribution.png')
#     plt.close()

# def create_samples_scaleFree(n=2000, m=1,
#                              iter_start=1,
#                              iter_end=15,
#                              alpha_range=np.arange(.0, 1, .05),
#                              save_fold=os.getcwd()
#                              ):
#     from auxiliary_scripts.scale_free.scale_free import scale_free
#     save_samples_dir = save_fold
#     ensure_dir(save_samples_dir)
#
#     print('\nCreating samples...\n')
#     total_iterations = len(range(iter_start, iter_end + 1)) * len(alpha_range)
#     progress_bar = tqdm(total=total_iterations)
#
#     for iter, alpha in tqdm(product(range(iter_start, iter_end + 1), alpha_range)):
#         G = scale_free(n=n, m=m, alpha=alpha)
#         nx.write_graphml(G, "{:s}/graph-r={:06.2f}-iter={:05d}.graphml".format(save_samples_dir, alpha, iter))
#         # Update progress bar
#         progress_bar.update(1)
#     # Close the progress bar
#     progress_bar.close()
#
#
# def create_dataset_optimization_model(load_fold: str, save_fold_prop: str, save_fold_ds: str=os.getcwd(), crt_properties: bool = True):
#     if crt_properties:
#         print('\nCreating dataset of properties...\n')
#         dict_files = order_files_by_r(file_list=sorted(glob.glob("{:s}/*.graphml".format(load_fold))))
#         ensure_dir(save_fold_prop)
#         # samples = []
#         for key, file_names in tqdm(dict_files.items()):
#             for file_name in file_names:
#                 G = nx.read_graphml(file_name)
#                 if not nx.is_connected(G):
#                     print(key)
#                 pickle_save(filename=save_fold_prop + file_name.rsplit('/', 1)[-1].replace('.graphml', '.pkl'),
#                             obj=dict_of_properties(G, key=key))
#                 # samples.append(dict_of_properties(G, key=key))
#
#     file_list = glob.glob(save_fold_prop + '*')
#     samples = []
#     for f in file_list:
#         iter = int(f.split('iter=')[1].split('.')[0])
#         sam = pickle_load(filename=f)
#         sam['iter'] = iter
#         # if (sam['n_nodes'] != 2000 or sam['n_edges'] != 2000 or
#         # if sam['number_of_spanning_trees'] < .5:
#         if np.max(sam['spectrum']) > 1e6:
#             print(f)
#         try:
#             samples.append(sam)
#         except:
#             print(f)
#     df = pd.DataFrame(samples)
#     # Save DataFrame to HDF5
#     ensure_dir(save_fold_ds)
#     df.to_hdf(f'{save_fold_ds}/dataset_nw.h5', key='data', mode='w')
