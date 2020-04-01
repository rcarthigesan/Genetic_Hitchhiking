"""
Created on Thu 03 Jan 15:34:10 2019

@author: R Carthigesan

Script used to calculate dynamics of mutation frequencies subject to drift, mutation and hitchhiking.
"""

import sim_methods as sim
import tqdm
# import matplotlib.pyplot as plt
import numpy as np

# Define Parameters

# parent_dir = "C:/Users/raman/OneDrive/Work/Cambridge_MASt_Physics/Project/Python/Genetic_Hitchhiking/Results/"
parent_dir = ""
test_name = "06_02_2019_no_growth"

mu = 1e-8  # 1e-8 is the value given by bionumbers
gamma = 0
n_init = int(5e4)
t_bottleneck = int(1e4)
growth_time = 164

n_sites = int(5e5)  # number of sites to model dynamics for (i.e. number of simulations)
n_gen = int(1.2e4)  # number of generations to model dynamics for
time_step = 1  # dynamics are calculated every time_step generations
times = np.arange(0, n_gen + time_step, time_step)

sampling_period = 100  # frequency is sampled every sampling_period time steps
sample_times = times[::sampling_period]

n_samples = len(sample_times)
results = np.zeros((n_samples, n_sites + 1))  # create extra row at end for time values
results[:, -1] = sample_times

cdf = sim.hitch_kern()  # generate hitchhiking kernel

# Calculate dynamics

for i in tqdm.tqdm(range(n_sites), desc="Calculating dynamics. Please wait..."):
    results[:, i] = sim.dynamics(times=times, dt=time_step, n_init=n_init, mu=mu, gamma=gamma, hitchhiking_kernel=cdf,
                                 sampling_period=sampling_period, t_bottleneck=t_bottleneck, growth_time=growth_time)
print("Dynamics calculated.")
print()
print()

# Save results and parameter values

print("Saving results...")
parameters = "N_init = " + str(n_init) + ", t_bottleneck = " + str(t_bottleneck) + ", growth_time = " +\
             str(growth_time) + ", Gamma = " + str(gamma) + ", mu = " + str(mu) + ", n_gen = " + str(n_gen) +\
             ", n_sites = " + str(n_sites) + ", dt = " + str(time_step) + ", sampling period = " + str(sampling_period)

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
