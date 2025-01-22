import powerlaw
from matplotlib import cm
from helpers import utils
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from helpers.visual_utils import set_ticks_label, set_legend, get_set_larger_ticks_and_labels, scientific_notation


def find_power_law_exponent(x, y):
    # Remove zero or negative values as log is not defined for these values
    x = np.array(x)
    y = np.array(y)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    # Take the logarithm of x and y
    log_x = np.log(x)
    log_y = np.log(y)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

    # The slope is the exponent alpha
    alpha = slope
    k = np.exp(intercept)

    return alpha, k


def plot_log_log_distribution(size_cc,
                              N: list,
                              x_lim_fit=None,
                              fig_format: str = 'png',
                              cmap_: str = 'Blues',
                              save_dir=None,
                              fig_name: str = 'dist_size_cc',
                              figsize: tuple = (6, 8),
                              show: bool = True,
                              grid: bool = False,
                              xticks_num: int = 5,
                              x_lim=None,
                              y_lim=None,
                              title_leg=None,
                              legend_flag: bool=True,
                              make_fit_flag: bool=False,
                              return_exponent: bool = False
                              ):
    cmap = cm.get_cmap(cmap_, len(np.unique(N))+1)

    # max_value = max(utils.unroll_nested_list(size_cc))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    for i, s in enumerate(size_cc):
        s = s[1:] # skip eigenvalue lambda=0
        fit1 = powerlaw.Fit(s, xmin=np.min(s)*.9, xmax=1.1 * np.max(s), discrete=True, density=True, fit_method='KS')
        binsc, probc = fit1.pdf(linear_bins=False, discrete=True, density=True)
        binsc = (binsc[1:] + binsc[:-1]) / 2.0
        probc[probc == 0] = 'nan'

        x_val = binsc
        y_val = probc

        c = np.argwhere(np.unique(N) == N[i])[0][0]

        # mask = (x_val > 0) & (y_val > 0)
        ax.plot(x_val, y_val, '-', markersize=4, label=f'{N[i]:.2f}', c=cmap(c))

        # Make fit
        if make_fit_flag:
            if x_lim_fit is not None:
                print('Fitting in the range: ', x_lim_fit)
                index_fit = (x_val > x_lim_fit[0]) & (x_val < x_lim_fit[1])
                slope, intercept = find_power_law_exponent(x_val[index_fit], y_val[index_fit])
            else:
                slope, intercept = find_power_law_exponent(x_val, y_val)

            ax.plot(x_val, intercept * np.array(x_val) ** slope, '--', markersize=3, c=cmap(c), alpha=.6)
            ax.text(0.05, 0.5 - i * .1, r'$\gamma$' + f'(r={N[i]:.2f})= {slope:.3f}',
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    # color='black')
                    color=cmap(c))

    # Set the scale of both axes to logarithmic
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set labels for axes
    ax.set_xlabel(r'$\mathbf{\lambda}$')
    ax.set_ylabel(r'$\mathbf{\mathcal{P}(\lambda)}$')

    if y_lim:
        ax.set_ylim(y_lim) #, bottom=1e-12)
    get_set_larger_ticks_and_labels(ax=ax)

    if legend_flag:
        set_legend(ax=ax,
                   title=title_leg if title_leg is not None else '',
                   loc=4, ncol=1)
    if grid:
        ax.grid()
    plt.tight_layout()
    if save_dir:
        plt.savefig(f'{save_dir}{fig_name}_phT.{fig_format}', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    if return_exponent:
        return slope

