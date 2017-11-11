import numpy as np
from bootstrap_replicates import draw_bootstrap_replicates
from matplotlib import pyplot as plt

from statsmodels.distributions.empirical_distribution import ECDF
def ecdf(vec):
    the_ecdf = ECDF(vec)
    return the_ecdf.x, the_ecdf.y

def ecdf_WRONG(vec):
    """plt.plot(vec_sorted, empirical_cdf, marker='.', linestyle='none')"""
    hist, bin_edges = np.histogram(vec, normed=True,
                               bins=len(vec)) #we do not want the counts here
    diff = bin_edges[1] - bin_edges[0]
    empirical_cdf = np.cumsum(hist) * diff
    return np.sort(vec), empirical_cdf


def plot_ecdf_bootstrap_replicates(size, random_state, vector, xlabel):
    cdfs = draw_bootstrap_replicates(ecdf, size, random_state, vector)

    for ii in range(len(cdfs)):
        cur_cdf = cdfs[ii]
        plt.plot(cur_cdf[0], cur_cdf[1], marker='.', linestyle='none',
                     color='gray',
                 alpha=0.1)

    #Compute and plot ECDF from original data
    x, y = ecdf(vector)
    plt.plot(x, y, marker='.')

    # Make margins and label axes
    plt.margins(0.02)
    plt.xlabel(xlabel)
    plt.ylabel('ECDF')
