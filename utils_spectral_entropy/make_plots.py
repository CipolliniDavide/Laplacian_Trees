import os
import matplotlib.pyplot as plt
import numpy as np
from helpers.visual_utils import set_ticks_label, set_legend, create_colorbar, get_set_larger_ticks_and_labels
# from utils_spectral_entropy.utils import find_peaks_indices
# from utils_spectral_entropy.spectral_entropy_numpy import specific_heat

def plot_curves(tau_range: np.array,
                           networks_eta: np.array,
                           r: np.array,
                y_label=r'$\mathbf{\eta}$',
                fig_name='eta',
                           save_dir: str = '{:s}'.format(os.getcwd()),
                           title: str = '', fig_format: str = 'png',
                           figsize: tuple = (7, 10),
                           tau_stars: np.array=None,
                           tau_lim: tuple = (5, 30),
                           num_xticks: int = 6,
                           number_of_curves: int = 6,
                           show=False
                           ):
    x_ticks = np.arange(0.1, .9, .1)
    valfmt_x = "{x:.1f}"

    fig = plt.figure(figsize=figsize, layout='tight')
    ax2 = plt.subplot(1, 1, 1)
    ax2 = plot_eta_curves(tau_range=tau_range,
                         networks_eta=networks_eta, r=r,
                         tau_stars=tau_stars,
                         tau_lim=tau_lim,
                         # cmap_name='viridis',
                          vline=False,
                          linewidth=2,
                         number_of_curves=number_of_curves,
                         ax=ax2,
                         num_xticks=num_xticks)
    ax2.grid(True, which='both', axis='x')
    ax2.set_ylabel(y_label)
    set_ticks_label(ax=ax2, ax_type='x',
                    # data=np.log10(tau_range),
                    data=x_ticks,
                    num=5,
                    valfmt=valfmt_x,
                    ticks=x_ticks,
                    only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                    ax_label=r'$\mathbf{r}$',
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                    add_ticks=[])
    plt.savefig('{:s}/curves_{:s}.{:s}'.format(save_dir, fig_name, fig_format), dpi=300)
    # get_set_larger_ticks_and_labels(ax=ax2, num_ticks_x=num_xticks)

    if show:
        plt.show()
    else:
        plt.close()



def plot_von_neumann_ent(von_neumann_ent, tau_range,
                         save_name=None, spec_heat=None,
                         tau_star_list=[],
                        figsize=(7, 5), labely='S', show=False, fig_format='.pdf',
                         legend_title=None, take_average=False,
                         ylim_ax2=(-.01, 1.5)):


    fig, ax1 = plt.subplots(figsize=figsize, tight_layout=True)  # Change layout to tight_layout
    if take_average:
        std_dev = von_neumann_ent.std(axis=1)
        von_neumann_ent_to_plot = von_neumann_ent.mean(axis=1)
        # Add shaded area representing standard deviation
        ax1.fill_between(tau_range, von_neumann_ent_to_plot - std_dev, von_neumann_ent_to_plot + std_dev, alpha=0.3)


    # Plot von Neumann entropy on the primary y-axis
    ax1.semilogx(tau_range, von_neumann_ent_to_plot, linewidth=2)  # , label=labely)
    ax1.set_xlabel(r'$\mathbf{\tau}$')
    get_set_larger_ticks_and_labels(ax=ax1)
    set_ticks_label(ax=ax1, ax_type='y', num=3, ax_label=labely, data=von_neumann_ent_to_plot)
    ax1.set_ylim(bottom=-.01, top=1.05)
    if spec_heat is not None:
        # spec_heat = specific_heat(spec_ent=von_neumann_ent_to_plot, tau_range=tau_range, batch_dim=1)
        # Create a second y-axis on the left for specific heat
        ax2 = ax1.twinx()
        # ax1.semilogx(tau_range[:-1], spec_heat, color='red', linewidth=2)
        if take_average:
            std_dev = spec_heat.std(axis=1)
            spec_heat = spec_heat.mean(axis=1)
            # Add shaded area representing standard deviation
            ax2.fill_between(tau_range, spec_heat - std_dev, spec_heat + std_dev, alpha=0.3, color='red')
        ax2.semilogx(tau_range, spec_heat, color='red', linewidth=2)
        # tau_of_lastpeak_list = tau_range[find_peaks_indices(x_array=tau_range, y_array=spec_heat, eps=eps_tau_star, atol=atol)]
        # for t in tau_star_list:
        for t, c in zip(tau_star_list, ['green', 'blue']):
            ax2.axvline(t, color='purple', linestyle='--', linewidth=3, alpha=.7, c=c)
            # ax3.text(t, ax1.get_ylim()[0], f"{t:.0f}", va='top', ha='center', color='gray')  # Add text box
        # ax2.tick_params(axis='y', labelcolor='red', color='red')
        ax2.set_ylabel(r'$\mathbf{C_{\tau}}$', color='red')
        num_ticks_y = 3
        # get_set_larger_ticks_and_labels(ax=ax2, num_ticks_y=num_ticks_y)
        set_ticks_label(ax=ax2, ax_type='y', num=num_ticks_y, data=spec_heat, ax_label=r'$\mathbf{C_{\tau}}$', color_label='red',
                        ticks=[.5, .66, 1])
        ax2.set_ylim(bottom=ylim_ax2[0], top=ylim_ax2[1])

        # ax.legend(loc='upper left')
        # Create top axis
        # ax3 = ax.twinx()  # Use twiny to create a new x-axis that shares the same y-axis with ax1
        # ax3.set_ylim(ax.get_ylim())  # Set the limits to match ax1
        # ax3.set_xticks(tau_stars)  # Add ticks corresponding to vertical lines
        # ax3.xaxis.tick_top()  # Move ticks to the top
        # ax3.tick_params(axis='x', labelcolor='gray')  # Adjust tick color

    # plt.title(tau_star)
    ax1.grid(True, which='both', axis='x')
    set_legend(ax=ax1, title=legend_title)

    # plt.show()
    # plt.close()

    if save_name:
        plt.savefig('{:s}.{:s}'.format(save_name, fig_format), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    a=0

def plot_thermo_trajectory(tau_range: np.array,
                           networks_eta: np.array,
                           r: np.array,
                           save_dir: str = '{:s}'.format(os.getcwd()),
                           title: str = '', fig_format: str = 'png',
                           figsize: tuple = (7, 10),
                           tau_stars: np.array=None,
                           tau_lim: tuple = (5, 30),
                           cmap_name: str = 'viridis',
                           num_xticks: int = 6,
                           number_of_curves: int = 6,
                           show=False
                           ):

    fig = plt.figure(figsize=figsize, layout='tight')
    plt.suptitle(title)
    ax1 = plt.subplot(2, 1, 1)
    # x, y = np.meshgrid(r, np.log10(tau_range))
    # map = ax1.pcolormesh(x, y, networks_eta, cmap=cmap_name)
    tau_mask = np.where((tau_range >= tau_lim[0]) & (tau_range <= tau_lim[1]))
    x, y = np.meshgrid(r, np.log10(tau_range[tau_mask]))
    map = ax1.pcolormesh(x, y, networks_eta[tau_mask], cmap=cmap_name)
    ax1.set_xticks([])
    set_ticks_label(ax=ax1, ax_type='y',
                    # data=np.log10(tau_range),
                    data=np.log10(tau_range[tau_mask]),
                    num=5,
                    valfmt="{x:.1f}",
                    ticks=None,
                    only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                    ax_label=r'$\mathbf{\log_{10}{\tau}}$',
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                    add_ticks=[])
    # set_ticks_label(ax=ax1, ax_type='x',
    #                 # data=np.log10(tau_range),
    #                 data=r,
    #                 num=9,
    #                 valfmt="{x:.2f}",
    #                 ticks=r,
    #                 only_ticks=False, tick_lab=None,
    #                 fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
    #                 ax_label=r'$\mathbf{r}$',
    #                 fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
    #                 add_ticks=[])
    get_set_larger_ticks_and_labels(ax=ax1, num_ticks_x=num_xticks)

    create_colorbar(fig=fig, ax=ax1, mapp=map,
                    array_of_values=networks_eta[tau_mask],
                    # array_of_values=networks_eta,
                    valfmt="{x:.2f}",
                    fontdict_cbar_label={'label': r'$\mathbf{\eta}$'},
                    fontdict_cbar_tickslabel=None, fontdict_cbar_ticks=None, position='right')

    ax2 = plt.subplot(2, 1, 2)
    ax2 = plot_eta_curves(tau_range=tau_range,
                    networks_eta=networks_eta, r=r,
                    tau_stars=tau_stars,
                    tau_lim=tau_lim,
                    # cmap_name='viridis',
                    number_of_curves=number_of_curves,
                    ax=ax2,
                    num_xticks=num_xticks)
    ax2.grid(True, which='both', axis='x')
    # get_set_larger_ticks_and_labels(ax=ax2, num_ticks_x=num_xticks)

    plt.savefig('{:s}/eta.{:s}'.format(save_dir, fig_format), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_thermo_trajectory_separate(tau_range: np.array,
                                    networks_eta: np.array,
                                    r: np.array,
                                    save_dir: str = '{:s}'.format(os.getcwd()),
                                    title: str = '', fig_format: str = 'png',
                                    figsize: tuple = (7, 10),
                                    tau_stars: np.array=None,
                                    tau_lim: tuple = (5, 30),
                                    cmap_name: str = 'viridis',
                                    cmap_curves_name: str = 'Spectral',
                                    linewidth: float = 4,
                                    num_xticks: int = 6,
                                    vline=True,
                                   number_of_curves: int = 6,
                                   show: bool=False,
                                   vmax=None, vmin=None,
                                   fig_name: str = 'eta',
                                   y_label: str = r'$\mathbf{\eta}$',
                                   add_yticks_heat=[],
                                   heat_num_yticks=5,
                                   valfmt_cbar="{x:.2f}",
                                   x_ticks=np.arange(0.1, .9, .1),
                                   valfmt_x = "{x:.1f}",
                                    valfmt_y="{x:.1f}",
                                    cbar_ticks=None,
                                    legend_loc=4
                                            ):

    # x_ticks = np.arange(0.1, .9, .1)
    # valfmt_x = "{x:.1f}"

    ######################################## Heat map #######################################################
    fig = plt.figure(figsize=figsize, layout='tight')
    plt.suptitle(title)
    ax1 = plt.subplot(1, 1, 1)
    # x, y = np.meshgrid(r, np.log10(tau_range))
    # map = ax1.pcolormesh(x, y, networks_eta, cmap=cmap_name)
    tau_mask = np.where((tau_range >= tau_lim[0]) & (tau_range <= tau_lim[1]))
    x, y = np.meshgrid(r, np.log10(tau_range[tau_mask]))
    map = ax1.pcolormesh(x, y, networks_eta[tau_mask], cmap=cmap_name, vmin=vmin, vmax=vmax)
    ax1.set_xticks([])
    set_ticks_label(ax=ax1, ax_type='y',
                    # data=np.log10(tau_range),
                    data=np.log10(tau_range[tau_mask]),
                    num=heat_num_yticks,
                    add_ticks=add_yticks_heat,
                    valfmt=valfmt_y,
                    ticks=None,
                    only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                    ax_label=r'$\mathbf{\log_{10}{\tau}}$',
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                    )
    set_ticks_label(ax=ax1, ax_type='x',
                    # data=np.log10(tau_range),
                    data=x_ticks,
                    num=5,
                    valfmt=valfmt_x,
                    ticks=x_ticks,
                    only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                    ax_label=r'$\mathbf{r}$',
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                    add_ticks=[])
    # get_set_larger_ticks_and_labels(ax=ax1, num_ticks_x=num_xticks)
    create_colorbar(fig=fig, ax=ax1, mapp=map,
                    array_of_values=np.clip(networks_eta[tau_mask], a_max=vmax, a_min=networks_eta[tau_mask].min()),
                    # array_of_values=networks_eta,
                    valfmt=valfmt_cbar,
                    cbar_edg_ticks=cbar_ticks,
                    fontdict_cbar_label={'label': y_label},
                    fontdict_cbar_tickslabel=None, fontdict_cbar_ticks=None, position='right')
    plt.savefig('{:s}/heatmap_{:s}.{:s}'.format(save_dir, fig_name, fig_format), dpi=300)


    ######################################## Plot curves #######################################################
    fig = plt.figure(figsize=figsize, layout='tight')
    ax2 = plt.subplot(1, 1, 1)
    ax2 = plot_eta_curves(tau_range=tau_range,
                         networks_eta=networks_eta,
                          r=r,
                          vline=vline,
                         tau_stars=tau_stars,
                         tau_lim=tau_lim,
                          linewidth=linewidth,
                         cmap_name=cmap_curves_name,
                         number_of_curves=number_of_curves,
                         ax=ax2,
                          legend_loc=legend_loc,
                         num_xticks=num_xticks)
    ax2.grid(True, which='both', axis='x')
    ax2.set_ylabel(y_label)
    set_ticks_label(ax=ax2, ax_type='x',
                    # data=np.log10(tau_range),
                    data=x_ticks,
                    num=5,
                    valfmt=valfmt_x,
                    ticks=x_ticks,
                    only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                    ax_label=r'$\mathbf{r}$',
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                    add_ticks=[])
    plt.savefig('{:s}/curves_{:s}.{:s}'.format(save_dir, fig_name, fig_format), dpi=300)
    # get_set_larger_ticks_and_labels(ax=ax2, num_ticks_x=num_xticks)
    if show:
        plt.show()
    else:
        plt.close()



def plot_eta_curves(tau_range: np.array,
                    networks_eta: np.array,
                    r: np.array,
                    ax=None,
                    # save_dir: str= '{:s}'.format(os.getcwd()),
                    title: str='',
                    # fig_format: str='png',
                    figsize: tuple=(7, 6),
                    num_xticks: int=6,
                    linewidth = 4,
                    tau_stars: np.array=None,
                    tau_lim: tuple=None,
                    cmap_name='Spectral',
                    legend_loc=4,
                    number_of_curves: int=6,
                    ylabel: str=r'$\mathbf{\eta}$',
                    vline: bool=True) -> None:

    if ax:
        pass
    else:
        fig = plt.figure(figsize=figsize, layout='tight')
        plt.suptitle(title)
        ax = plt.subplot(1, 1, 1)

    # Select num indices of tau that satisfy the condition
    # Define the range condition for tau
    if tau_lim and (tau_stars is None):
        tau_condition = (tau_range > tau_lim[0]) & (tau_range <= tau_lim[1])
        I = np.where(tau_condition)[0]
        I = I[::len(I) // (number_of_curves - 1)]
    elif tau_stars is not None:
        tau_condition = np.isin(tau_range, tau_stars)
        I = np.where(tau_condition)[0]
    else:
        I = np.arange(0, networks_eta.shape[0], 1)
        tau_condition = np.ones_like(tau_range, dtype=bool)

    # Get the corresponding tau values and colors
    # selected_tau_values = tau_range[I]
    # cmap_name = plt.get_cmap('viridis_r')
    cmap_name = plt.get_cmap(cmap_name)

    slicedCM = cmap_name(np.linspace(0.1, .9, len(I)))


    for ind, color in zip(I, slicedCM):
        tau = tau_range[ind]
        label = scientific_notation(tau)
        ax.plot(r, networks_eta[ind],
                    label=label,
                    color=color, linewidth=linewidth)
        if vline:
            ax.axvline(r[np.argmax(networks_eta[ind])], color=color, linestyle='--', linewidth=linewidth, alpha=.6)
        # print(r'tau={:.1f}, r_max={:.1f}'.format(tau, r[np.argmax(networks_eta[ind])]))

    set_legend(ax=ax, loc=legend_loc, title=r'$\mathbf{\tau}$', ncol=2)
    # ax.set_ylim(bottom=0, top=.6)
    ax.set_xlabel(r'$\mathbf{r}$')
    # get_set_larger_ticks_and_labels(ax=ax, num_ticks_x=num_xticks)
    # print(num_xticks)
    set_ticks_label(ax=ax, ax_type='y',
                    data=networks_eta[tau_condition].reshape(-1),
                    # data=[0, .45],
                    num=5, valfmt="{x:.2f}", ticks=None,
                    only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                    ax_label=ylabel,
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                    add_ticks=[])

    set_ticks_label(ax=ax, ax_type='x', data=r, num=10, valfmt="{x:.2f}", ticks=r[::2],
                    only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                    ax_label=r'$\mathbf{r}$',
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                    add_ticks=[])

    return ax


def plot_bothefficiency(r_list, tau_range, eta_mean, alg_eta_mean, tau_index, legend_loc=1, fig_format='png',
                        figsize: tuple=(12, 6), save_dir: str='./', top_ylim=None,
                        show=False):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, layout='tight', figsize=figsize)
    for index in tau_index:

        tau = tau_range[index]
        label = scientific_notation(tau)
        for y_value, c, y_lab, ax in [
                                      (eta_mean[:, index], 'red', r'$\mathbf{\eta_{Therm}}$', ax1),
                                      # (eta_mean, 'red', r'$\mathbf{\eta_{Therm}}$', ax1),
                                      (alg_eta_mean, 'blue', r'$\mathbf{\eta_{Alg}}$', ax2)
                                      # (dS_mean[:, index], 'purple', r'$\mathbf{dS}$', ax2),
                                      # (dF_mean[:, index], 'red', r'$\mathbf{dF}$', ax3)
                                      ]:
            ax.plot(r_list, y_value, linewidth=3.5, alpha=.6,
                    label=label
                    )
            set_ticks_label(ax=ax, ax_type='y', ax_label=y_lab, data=y_value[1:], valfmt="{x:.2f}")
            set_ticks_label(ax=ax, ax_type='x', ax_label='r', data=r_list, ticks=r_list[::2], valfmt="{x:.2f}")
    # [ax.grid() for ax in [ax1, ax2, ax3]]
    [ax.grid() for ax in [ax1, ax2]]
    if top_ylim is not None:
        ax1.set_ylim(top=top_ylim, bottom=-.01)
    set_legend(ax=ax1, title=r'$\mathbf{\tau}$', ncol=2, loc=legend_loc)
    plt.savefig(f'{save_dir}/efficiency.{fig_format}', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_eta_dS_dF(eta_mean: np.array, dF_mean: np.array, dS_mean: np.array, tau_range: np.array,
                   tau_index: np.array, r_list: np.array, fig_format: str='png', save_dir: str='',
                   show=False, legend_loc=4, top_ylim=.4, fig_name='eta_dS_dF',
                   ylabels=[r'$\mathbf{\eta}$', r'$\mathbf{\delta S}$', r'$\mathbf{\delta F}$'],
                   ncol_leg=3,
                   x_ticks=None,
                   ):

    if x_ticks is None:
        x_ticks = r_list[::2]
    else:
        pass

    colormap = plt.cm.magma
    # Adjust the number of colors in the colormap to match the number of curves
    colors = [colormap(i) for i in np.linspace(0, 1, len(tau_index))]
    # Set the color cycle to use the chosen colors
    # plt.gca().set_prop_cycle('color', colors)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=False, layout='tight', figsize=(20, 6))
    for i, index in enumerate(tau_index):

        tau = tau_range[index]
        label = scientific_notation(tau)
        for data, c, y_lab, ax in [(eta_mean, 'red', ylabels[0], ax1),
                                    (dS_mean, 'purple', ylabels[1], ax2),
                                    (dF_mean, 'red', ylabels[2], ax3)]:
            y_value = data[:, index]
            ax.plot(r_list, y_value, linewidth=3.5, alpha=.6, label=label, color=colors[i])
            set_ticks_label(ax=ax, ax_type='y', ax_label=y_lab, data=data[:, tau_index], valfmt="{x:.2f}")
            set_ticks_label(ax=ax, ax_type='x', ax_label='r', data=r_list, ticks=x_ticks, valfmt="{x:.1f}")
    [ax.grid() for ax in [ax1, ax2, ax3]]
    # ax1.set_ylim(top=top_ylim, bottom=-.01)
    set_legend(ax=ax1, title=r'$\mathbf{\tau}$', ncol=ncol_leg, loc=legend_loc)
    plt.savefig(f'{save_dir}{fig_name}.{fig_format}', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

# def scientific_notation(number: float) -> str:
#     exponent = int(np.floor(np.log10(number)))
#     mantissa = number / 10 ** exponent
#     if exponent >= 0:
#         label = '${:.1f} \\times 10^{{{}}}$'.format(mantissa, exponent)
#     else:
#         label = '${:.1f} \\times 10^{{{}{}}}$'.format(mantissa, "" if mantissa < 0 else " ", exponent),
#     return label

def scientific_notation(number: float) -> str:
    exponent = int(np.floor(np.log10(np.abs(number))))
    mantissa = number / 10 ** exponent
    label = '${:.1f} \\times 10^{{{}}}$'.format(mantissa, exponent)
    return label

def plot_eta_accuracy(eta_mean: np.array,
                      accuracy: np.array,
                      #dF_mean: np.array, dS_mean: np.array,
                      tau_range: np.array,
                      tau_index: np.array, r_list: np.array, fig_format: str='png', save_dir: str='',
                      show=False, legend_loc=4, top_ylim=.4):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=False, layout='tight', figsize=(20, 6))
    for index in tau_index:

        tau = tau_range[index]
        label = scientific_notation(tau)
        for y_value, c, y_lab, ax in [(eta_mean[:, index], 'red', r'$\mathbf{\eta}$', ax1),
                                    #(dS_mean[:, index], 'purple', r'$\mathbf{dS}$', ax2),
                                     # (dF_mean[:, index], 'red', r'$\mathbf{dF}$', ax3)
                                      ]:
            ax.plot(r_list, y_value, linewidth=3.5, alpha=.6, label=label)
            set_ticks_label(ax=ax, ax_type='y', ax_label=y_lab, data=y_value[1:], valfmt="{x:.2f}")
            set_ticks_label(ax=ax, ax_type='x', ax_label='r', data=r_list, ticks=r_list[::2], valfmt="{x:.2f}")
    [ax.grid() for ax in [ax1, ax2, ax3]]
    ax1.set_ylim(top=top_ylim, bottom=-.01)
    set_legend(ax=ax1, title=r'$\mathbf{\tau}$', ncol=2, loc=legend_loc)
    plt.savefig(f'{save_dir}efficiency.{fig_format}', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_surface(z,
                 r,
                 tau_range,
                 save_dir: str='./',
                 show: bool=False,
                 z_label: str=r'$\mathbf{S}$',
                 z_lim: tuple=(None, None),
                 legend_loc=4,
                 tau_lastpeak=None,
                 tau_firstpeak=None,
                 z_curve1=None, z_curve2=None,
                 z_firstpeak=None,
                 z_lastpeak=None,
                 title: str='',
                 view_init=(30, 60),
                 alpha=.5,
                 vmax=None,
                 cmap='viridis',
                 label_overlayed_curve=(r'$C^{(1)}_{peak}$', r'$C^{(2)}_{peak}$'),
                 fig_format: str='png'):

    if vmax is None:
        vmax=np.max(z)

    x, y = np.meshgrid(r, np.log(tau_range))
    from mpl_toolkits.mplot3d import Axes3D

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(x, y, z, cmap=cmap,
                           alpha=alpha,
                           vmax=vmax,
                           linewidth=.5,
                           # antialiased=False,
                           rstride=1, cstride=1
                           )

    ax.plot(r, np.log(tau_firstpeak), z_firstpeak, color='blue', label=label_overlayed_curve[0], linewidth=3, alpha=1,
            marker='^')
    ax.plot(r, np.log(tau_lastpeak), z_lastpeak, color='green', label=label_overlayed_curve[1], linewidth=3, alpha=1,
            marker='o')

    # Find intersection point
    # matching_indices = np.where(np.array(tau_lastpeak) == np.array(tau_firstpeak))[0][-1]
    # ax.scatter(r_[matching_indices], np.log(tau_lastpeak[matching_indices]), critical_entropy_lastpeak[matching_indices], marker='o',
    #            color='black', alpha=1,
    #            label='({:.2f}-{:.2f}, {:.2f}-{:.2f}, {:.2f}-{:.2f})'.format(r_[matching_indices], r_[matching_indices+1],
    #                                                                         tau_lastpeak[matching_indices], tau_lastpeak[matching_indices+1],
    #                                                                         critical_entropy_lastpeak[matching_indices], critical_entropy_lastpeak[matching_indices+1]))
    # # ax.scatter(r_[matching_indices+1], np.log(tau_lastpeak[matching_indices+1]),
    #            critical_entropy_lastpeak[matching_indices+1], marker='o',
    #            color='black', alpha=1,)
    # Add labels
    ax.set_xlabel('$\mathbf{r}$', fontsize=18)
    ax.set_ylabel(r'$\mathbf{\tau}$', fontsize=18)
    ax.set_zlabel(z_label, fontsize=18)

    ax.xaxis._axinfo['juggled'] = (0, 0, 0)
    ax.yaxis._axinfo['juggled'] = (1, 1, 1)
    ax.zaxis._axinfo['juggled'] = (2, 2, 2)
    ax.view_init(view_init[0], view_init[1])

    # Set y-axis to log scale
    # ax.set_yscale('log')
    # Add a color bar which maps values to colors.
    # fig.colorbar(surf)
    # ax.view_init(20, -320)
    if z_lim[0]:
        ax.set_zlim(bottom=z_lim[0])
    if z_lim[1]:
        ax.set_zlim(top=z_lim[1])

    plt.legend(fontsize="15")
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{title}_surf.{fig_format}', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_tau_vs_r(r_, tau_lastpeak, tau_firstpeak, spectrum, show: bool=False, save_dir: str='', fig_format: str='png',
                  y_lim=None):
    tau_of_lastpeak_list = np.array(tau_lastpeak)
    # Compute the average first non-zero eigenvalue of the Laplacian
    min_lambda = np.array([np.sort(spectrum[:, i].squeeze(), axis=1)[:, 1] for i in range(spectrum.shape[1])])
    tau_diff_avg = (1 / min_lambda).mean(1)
    tau_diff_std = (1 / min_lambda).std(1)

    figsize = (7, 5)
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)  # Change layout to tight_layout
    ax.scatter(r_, np.array(tau_firstpeak), label=r'$\tau^{(1)}_{peak}$', c='blue', alpha=1, marker='x', s=15)
    ax.scatter(r_, tau_of_lastpeak_list, label=r'$\tau^{(2)}_{peak}$', c='green', alpha=.6, marker='o', s=45)
    ax.errorbar(r_, tau_diff_avg,
                # markersize=8,
                yerr=[tau_diff_std*1, tau_diff_std],
                # yerr=(1/min_lambda).std()*1.96/np.sqrt(len(min_lambda)),
                fmt='v',
                elinewidth=1.5,
                capsize=5,
                label=r'$\tau_{diff}=1/\lambda_{2}$',
                alpha=.4,
                c='red')
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\mathbf{\tau}$')
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    get_set_larger_ticks_and_labels(ax=ax, num_ticks_x=10)
    set_legend(ax=ax)
    plt.grid()
    plt.savefig(f'{save_dir}/tau_vs_r.{fig_format}')
    if show:
        plt.show()
    else:
        plt.close()

    figsize = (6, 5)
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)  # Change layout to tight_layout
    # ax.scatter(r_, np.array(tau_firstpeak), label=r'$\tau^{(1)}_{peak}$', c='blue', alpha=1, marker='x', s=15)
    # ax.scatter(r_, tau_of_lastpeak_list, label=r'$\tau^{(2)}_{peak}$', c='green', alpha=.6, marker='o', s=45)
    ax.scatter(r_, tau_diff_std**2, label=r'$\tau_{diff}=1/\lambda_{2}$',
                 c='red', alpha=.6, marker='o', s=45)
    # ax.errorbar(r_, tau_diff_avg,
    #             # markersize=8,
    #             yerr=[tau_diff_std * 1, tau_diff_std],
    #             # yerr=(1/min_lambda).std()*1.96/np.sqrt(len(min_lambda)),
    #             fmt='v',
    #             elinewidth=1.5,
    #             capsize=5,
    #             label=r'$\tau_{diff}=1/\lambda_{2}$',
    #             alpha=.4,
    #             c='red')
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\mathbf{Var(\tau_{diff})}$')
    # if y_lim:
    #     ax.set_ylim(y_lim[0], y_lim[1])
    get_set_larger_ticks_and_labels(ax=ax, num_ticks_x=10)
    # set_legend(ax=ax)
    plt.grid()
    plt.savefig(f'{save_dir}/taudiff_std_vs_r.{fig_format}')


def plot_quantity_along_transition(quantity,
                                   r_: np.array,
                                   std_quantity=None,
                                   ylabel_std_quantity=r'$\mathbf{N \cdot Var}$',
                                   figsize: tuple = (6, 5),
                                   ylabel: str = 'S',
                                   fig_name: str = '',
                                   color: str = 'green',
                                   marker: str = 'o',
                                   save_dir: str = '', fig_format: str='png', marker_size: float=45, y_lim=(0, 1),
                                   x_ticks=None):
    if x_ticks is None:
        x_ticks = r_[::2]

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)  # Change layout to tight_layout
    ax.scatter(r_, quantity, #label=r'$\tau_{diff}=1/\lambda_{2}$',
               c=color, alpha=1, marker=marker, s=45)
    # ax.set_xlabel('r')
    # ax.set_ylabel(ylabel)
    # if y_lim:
    #     ax.set_ylim(y_lim[0], y_lim[1])
    set_ticks_label(ax=ax, ax_type='y', data=[y_lim[0], y_lim[1]], num=5, valfmt="{x:.2f}", #ticks=r[::2],
                    only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                    ax_label=ylabel,
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                    add_ticks=[])

    # get_set_larger_ticks_and_labels(ax=ax, num_ticks_x=10)
    set_ticks_label(ax=ax, ax_type='x', data=r_, num=10, valfmt="{x:.1f}", ticks=x_ticks,
                    only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                    ax_label=r'$\mathbf{r}$',
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                    add_ticks=[])
    if std_quantity is None:
        pass
    else:
        ax2 = ax.twinx()
        ax2.plot(r_, std_quantity, c=color, ls=':')
        set_ticks_label(ax=ax2, ax_type='y', data=std_quantity, num=3, valfmt="{x:.1f}",  # ticks=r[::2],
                        only_ticks=False,
                        tick_lab=None,
                        fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                        ax_label=ylabel_std_quantity,
                        fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                        add_ticks=[])
        ax.set_ylim(bottom=-.02)
        ax2.set_ylim(bottom=-2)
    # set_legend(ax=ax)
    ax.grid()
    plt.savefig(f'{save_dir}/{fig_name}.{fig_format}')


