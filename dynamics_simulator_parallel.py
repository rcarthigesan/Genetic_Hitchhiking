"""
Created on Thu 03 Jan 15:34:10 2019

@author: R Carthigesan

Script used to calculate dynamics of mutation frequencies subject to drift, mutation and hitchhiking.
"""

import sim_methods as sim
import numpy as np
# import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Define Parameters

parent_dir = ""
# parent_dir = "C:/Users/raman/OneDrive/Work/Cambridge_MASt_Physics/Project/Python/Genetic_Hitchhiking/Results/"


test_name = "05_03_2019_slow_low_1e-8"
n_cores = 16

mu = 1e-8
gamma = 0
n_init = int(5e4)
t_bottleneck = int(0)
growth_time = 16400

n_sites = int(5e5)  # number of sites to model dynamics for (i.e. number of simulations)
n_gen = int(1.02e5)  # number of generations to model dynamics for
time_step = 1  # dynamics are calculated every time_step generations
times = np.arange(0, n_gen + time_step, time_step)

sampling_period = 1000  # frequency is sampled every sampling_period time steps
sample_times = times[::sampling_period]

n_samples = len(sample_times)

cdf = sim.hitch_kern()  # generate hitchhiking kernel

# Calculate dynamics

inputs = list(range(n_sites))

print("Calculating dynamics for " + str(n_sites) + " sites. Please wait...")


def process(i):
    return sim.dynamics(times=times, dt=time_step, n_init=n_init, mu=mu, gamma=gamma, hitchhiking_kernel=cdf,
                        sampling_period=sampling_period, t_bottleneck=t_bottleneck, growth_time=growth_time)


results_list = Parallel(n_jobs=n_cores, verbose=50)(delayed(process)(i) for i in inputs)
results = np.array(results_list).transpose()
results = np.c_[results, sample_times]

print("Dynamics calculated.")
print()
print()

# Save results and parameter values

print("Saving results...")
parameters = "N_init = " + str(n_init) + ", t_bottleneck = " + str(t_bottleneck) + ", growth_time = " +\
             str(growth_time) + ", Gamma = " + str(gamma) + ", mu = " + str(mu) + ", n_gen = " + str(n_gen) +\
             ", n_sites = " + str(n_sites) + ", dt = " + str(time_step) + ", sampling period = " + \
             str(sampling_period)

if parent_dir != "":
    np.save(parent_dir + test_name, results)
    file = open(parent_dir + test_name + "_parameters.txt", "w")
    file.write(parameters)
    file.close()
    print("Results saved successfully to directory: " + parent_dir + " with name: " + test_name + ".")

else:
    np.save(test_name, results)
    file = open(test_name + "_parameters.txt", "w")
    file.write(parameters)
    file.close()
    print("Results saved successfully with name: " + test_name + ".")
print()
print()

# # Plot dynamics
#
# print("Plotting results...")
# for i in list(range(results.shape[1] - 1)):
#     plt.plot(results[:, -1], results[:, i])
# print("Results plotted.")
# plt.show()
