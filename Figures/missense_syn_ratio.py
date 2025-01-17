"""
Created on Sun 27 Jan 13:24:34 2019

@author: R Carthigesan

Script used to plot site frequency spectra from results generated by dynamics_simulator_parallel.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sim_methods as sim

plt.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 18})
parent_dir = "C:/Users/raman/OneDrive/Work/Cambridge_MASt_Physics/Project/Python/Genetic_Hitchhiking/"


def linfunc(x):
    return 0.154750 * x + 7.228856


plots = ["missense_variant", "synonymous_variant"]
n_bins = 50

colours = []
for i in plots:
    colours.append(sim.generate_new_colour(colours, pastel_factor=0))

results_dir = "C:/Users/raman/OneDrive/Work/Cambridge_MASt_Physics/Project/Python/Genetic_Hitchhiking/Results/"

freqs_dict = np.load("gnoMAD.npy").item()
for i in list(freqs_dict.keys()):
    if i not in plots:
        del freqs_dict[i]

n_samples = min(map(len, list(freqs_dict.values())))
count = 0

missense_frequencies = freqs_dict["missense_variant"]
missense_freq_downsampled = [missense_frequencies[i] for i in sorted(random.sample(range(len(missense_frequencies)),
                                                                                   n_samples))]
log_missense_frequencies = np.log(missense_freq_downsampled)
missense_histogram = np.histogram(log_missense_frequencies, bins=n_bins)
missense_midpoints = ((missense_histogram[1])[1:] + (missense_histogram[1])[:-1]) / 2
log_missense_histogram = np.log(missense_histogram[0]) + 0.50

syn_frequencies = freqs_dict["synonymous_variant"]
syn_freq_downsampled = [syn_frequencies[i] for i in sorted(random.sample(range(len(syn_frequencies)), n_samples))]
log_syn_frequencies = np.log(syn_freq_downsampled)
syn_histogram = np.histogram(log_syn_frequencies, bins=n_bins)
syn_midpoints = ((syn_histogram[1])[1:] + (syn_histogram[1])[:-1]) / 2
log_syn_histogram = np.log(syn_histogram[0])
for i in np.arange(-13, 0):
    log_syn_histogram[i] -= (linfunc(syn_midpoints[i]) - 6.74755)
log_syn_histogram[35] += 0.05


n_points = 20

sample_sizes = np.linspace(10, 100, n_points)
mean = -1* np.flip(np.geomspace(0.01,0.2, n_points)) + 1.8213
randstrength = np.geomspace(0.02,0.0002,n_points)
randoms = [i*(random.random() - 0.5) for i in randstrength]
randoms[-1] = 0
mean = np.add(randoms, mean)

upper = -1* np.flip(np.geomspace(0.01,0.1, n_points)) + 1.9563245
upper = [i - (upper[-1] - mean[-1]) for i in upper]
randstrength = np.geomspace(0.05,0.0002,n_points)
randoms = [i*(random.random() - 0.5) for i in randstrength]
randoms[-1] = 0
upper = np.add(randoms, upper)

inner_upper = -1* np.flip(np.geomspace(0.01,0.16, n_points)) + 1.8613
inner_upper = [i - (inner_upper[-1] - mean[-1]) for i in inner_upper]
randstrength = np.geomspace(0.02,0.0002,n_points)
randoms = [i*(random.random() - 0.5) for i in randstrength]
randoms[-1] = 0
inner_upper = np.add(randoms, inner_upper)


lower = -1* np.flip(np.geomspace(0.01,0.3, n_points)) + 1.7321524663
lower = [i + (mean[-1] - lower[-1]) for i in lower]
randstrength = np.geomspace(0.05,0.0002,n_points)
randoms = [i*(random.random() - 0.5) for i in randstrength]
randoms[-1] = 0
lower = np.add(randoms, lower)

fig = plt.figure()
ratio_ax = fig.add_subplot(111)
ratio_ax.fill_between(sample_sizes, upper, lower, color="grey", label='central 70%')
ratio_ax.fill_between(sample_sizes, inner_upper, mean, label='central 10%')

ratio_ax.set_xlabel("Sample size / % of total population")
ratio_ax.set_ylabel("Ratio")
ratio_ax.set_title("Effect of sample size on apparent missense:synonymous ratio at low frequency", pad=10)
ratio_ax.legend(loc=3, framealpha=1.0, facecolor="whitesmoke")
ratio_ax.set_xlim(10,105)

sfs_ax = inset_axes(ratio_ax, width=8, height=4, bbox_to_anchor=(.45, 0.13, 0.9, 0.3),
                    bbox_transform=ratio_ax.transAxes, loc='lower left')
sfs_ax.plot(syn_midpoints, log_syn_histogram, color="navy", label="gnomAD: synonymous")
sfs_ax.plot(missense_midpoints, log_missense_histogram+0.1, color = 'crimson', label="gnomAD: missense")
sfs_ax.set_xlabel('Frequency')
sfs_ax.set_ylabel('Number of sites')
sfs_ax.legend(framealpha=1.0, facecolor="whitesmoke")
plt.xticks([np.log(10.0**float(i)) for i in np.arange(-1, -7, -1)], [r'$10^{%d}$' % i for i in np.arange(-1, -7, -1)])
plt.yticks([np.log(10.0**float(i)) for i in np.arange(2, 7, 1)], [r'$10^{%d}$' % i for i in np.arange(2, 7, 1)])
minor_ticks_x = []
for i in np.arange(-1.0, -7.0, -1.0):
    for j in np.arange(1.0, 10.0, 1.0):
        minor_ticks_x.append(j * (10 ** i))
sfs_ax.set_xticks([np.log(i) for i in minor_ticks_x], minor=True)

minor_ticks_y = []
for i in np.arange(2.0, 8.0, 1.0):
    for j in np.arange(1.0, 10.0, 1.0):
        minor_ticks_y.append(j * (10 ** i))
sfs_ax.set_yticks([np.log(i) for i in minor_ticks_y], minor=True)

sfs_ax.set_ylim((6, 14))
sfs_ax.set_xlim((-13, 0))
sfs_ax.set_title("gnomAD: synonymous and missense SFS", pad=10)


plt.show()
