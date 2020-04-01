"""
Created on Sun 27 Jan 13:24:34 2019

@author: R Carthigesan

Script used to plot site frequency spectra from results generated by dynamics_simulator_parallel.py.
"""

# Import libraries and define useful plotting function

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import matplotlib
from matplotlib.widgets import Slider

plt.style.use('ggplot')
# matplotlib.rcParams.update({'font.size': 20})  # for full width figures
matplotlib.rcParams.update({'font.size': 25})  # for half-width figures

# Specify desired results and time at which to plot SFS

test_name = "mut_drift"
t_interest = int(200)  # time at which to plot SFS for simulations
consequences = []
n_bins = 100

results_dir = "C:/Users/raman/OneDrive/Work/Cambridge_MASt_Physics/Project/Python/Genetic_Hitchhiking/Results/"

params_file = open(results_dir + test_name + "_parameters.txt", "rt")
params_txt = params_file.read()
params_file.close()
sampling_period = int((params_txt.rsplit())[-1])
n_sites = int(((params_txt.rsplit())[-8])[:-1])
n_gen = int(((params_txt.rsplit())[-11])[:-1])
results = np.load(results_dir + test_name + ".npy")


def plotter(axis, time):
    frequencies = np.array((results[int(time / sampling_period), :])[0:-1])
    frequencies = frequencies[frequencies != 0]
    thresh_freqs = []
    for i in frequencies:
        if i != 1.0 and i != 0.0:
            thresh_freqs.append(i)
    logit_frequencies = special.logit(thresh_freqs)
    histogram = np.histogram(logit_frequencies, bins=n_bins)
    midpoints = ((histogram[1])[1:] + (histogram[1])[:-1]) / 2
    axis.bar(midpoints, histogram[0], log=True, width=(midpoints[1] - midpoints[0]), alpha=0.9)


fig = plt.figure()
plt.axis("off")
plt.title("SFS")
ls = []

ax = fig.add_subplot(111)
l = plotter(ax, t_interest)

labels = [0.01, 0.1, 0.5, 0.9, 0.99]
locs = [special.logit(i) for i in labels]

plt.legend()

ls.append(l)
plt.axis('off')

axsl = plt.axes([0.15, 0.00, 0.65, 0.03])
sframe = Slider(axsl, 'Time', 0, 200000, valinit=0)


def update(ax, val):
    frame = int(sampling_period * round(float(val)/sampling_period))
    ax.cla()
    l = plotter(ax, frame)
    ax.set_xticks(locs)
    ax.set_xticklabels(labels)
    ax.set_xlim(locs[0], locs[-1])
    ax.set_ylim(0.5, 500)
    ax.set_ylabel('Count')
    ax.set_xlabel('Frequency')
    plt.draw()


sframe.on_changed(lambda val: update(ax, val))
fig.subplots_adjust(bottom=0.2)
plt.show()