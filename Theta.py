"""
Created on 18/03/2019 12:38

@author: R Carthigesan

"""

# Import libraries and define useful plotting function

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
# from scipy.stats import linregress
import matplotlib
# import random
import sim_methods as sim

matplotlib.rcParams.update({'font.size': 18})


def sfs_no_growth(lnfreqs, N, mu, t):
    return np.log(N*mu) - ((N / t) * np.exp(lnfreqs))


def abline(slope, intercept, lab, col):

    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', label=lab, color=col)


def pop(t, n_init, t_bottleneck, growth_time):
    if t < t_bottleneck:
        return n_init
    else:
        return n_init * np.exp((t - t_bottleneck) / growth_time)


def sfs_growth(lnfreqs, t, mu, n_init, t_bottleneck, growth_time):
    N = pop(t, n_init, t_bottleneck, growth_time)
    freqs = np.exp(lnfreqs)

    def integrand(x, N, f, t, growth_time):
        return (1 / x**2) * np.exp(-N * f / t) * np.exp(x / growth_time)

    rho = []

    for f in freqs:
        rho.append(mu * n_init * (np.exp(- N * f / t) - np.exp(- N * f / (t - t_bottleneck)))\
              + f * mu * N**2 * integrate.quad(integrand, 0, t - t_bottleneck, args=(N, f, t, growth_time))[0])

    return np.log(rho)


def sfs_growth_2(lnfreqs, t, mu, n_init, t_bottleneck, growth_time):

    def N(t):
        return pop(t, n_init, t_bottleneck, growth_time)

    freqs = np.exp(lnfreqs)
    rho = []

    def integrand(x, n, f, t):
        return n(x) * (1 / (t - x)**2) * np.exp((-f * n(x)) / (t - x))

    for f in freqs:
        rho.append(f * mu * N(t) * integrate.quad(integrand, 0, t, args=(N, f, t))[0])

    return np.log(rho)


# Specify desired results and time at which to plot SFS

test_names = ["03_03_2019_high_init_even_longer_1e-8"]
t_interest = int(1.2e4)  # time at which to plot SFS for simulations
consequences = ["synonymous_variant", "missense_variant"]
afr_consequences = [('afr_' + i) for i in consequences]
n_bins = 50

plots = test_names + afr_consequences\
        + consequences

colours = []
for i in plots:
    colours.append(sim.generate_new_colour(colours, pastel_factor=0))

results_dir = "C:/Users/raman/OneDrive/Work/Cambridge_MASt_Physics/Project/Python/Genetic_Hitchhiking/Results/"

freqs_dict = np.load("gnoMAD.npy").item()

# Import results and sampling time

for test_name in test_names:
    params_file = open(results_dir + test_name + "_parameters.txt", "rt")
    params_txt = params_file.read()
    params_file.close()
    sampling_period = int((params_txt.rsplit())[-1])
    n_sites = int(((params_txt.rsplit())[-8])[:-1])
    results = np.load(results_dir + test_name + ".npy")
    if test_name == "12_02_2019_long_1e-8":
        frequencies = np.array((results[int(2.2e4 / sampling_period), :])[0:-1])
    elif test_name == "12_02_2019_longer_2e-8":
        frequencies = np.array((results[int(52000 / sampling_period), :])[0:-1])
    elif test_name == "12_02_2019_very_long_15e-9":
        frequencies = np.array((results[int(102000 / sampling_period), :])[0:-1])
    elif test_name == "13_02_2019_very_long_1e-8":
        frequencies = np.array((results[int(102000 / sampling_period), :])[0:-1])
    elif test_name == "01_03_2019_high_init_very_long_1e-8":
        frequencies = np.array((results[int(102000 / sampling_period), :])[0:-1])
    elif test_name == "03_03_2019_high_init_even_longer_1e-8":
        frequencies = np.array((results[int(502000 / sampling_period), :])[0:-1])
    elif test_name == "05_03_2019_slow_low_1e-8":
        frequencies = np.array((results[int(52000 / sampling_period), :])[0:-1])
    else:
        frequencies = np.array((results[int(t_interest / sampling_period), :])[0:-1])
    frequencies = frequencies[frequencies != 0]
    freqs_dict[test_name] = frequencies

n_samples = min(map(len, list(freqs_dict.values())))
count = 0

# theoretical_freqs = sim.randomvariate(theoretical_SFS, N=5e5, mu=1e-8, t=1.02e5, n_samples=n_samples, xmin=0, xmax=1)[0]

plt.figure()
plt.grid()

for plot in plots:
    frequencies = freqs_dict[plot]
    log_frequencies = np.log(frequencies)
    # freq_downsampled = [frequencies[i] for i in sorted(random.sample(range(len(frequencies)), n_samples))]
    # log_frequencies = np.log(freq_downsampled)
    histogram = np.histogram(log_frequencies, bins=n_bins)
    midpoints = ((histogram[1])[1:] + (histogram[1])[:-1]) / 2
    bin_widths = np.diff(histogram[1])
    if plot == "05_03_2019_slow_low_1e-8":
        midpoints += 3
    if "missense" in  plot:
        plt.semilogy(midpoints, np.divide(histogram[0], bin_widths * 1.4), marker='x', label=plot)
    elif "synonymous" in  plot:
        plt.semilogy(midpoints, np.divide(histogram[0], bin_widths * 0.7), marker='x', label=plot)
    else:
        plt.semilogy(midpoints, np.divide(histogram[0], bin_widths), marker='x', label=plot)
    count += 1


lnfreqs = np.linspace(-10, 0, n_bins)
# plt.plot(lnfreqs, sfs_no_growth(lnfreqs, N=1e10, mu=1.2e-7, t=6e9), linewidth=5, alpha=0.3, color='b', label='theoretical_no_growth')
# plt.plot(lnfreqs, sfs_growth(lnfreqs, t=1.02e5, mu=1e-8, n_init=5e4, t_bottleneck=1e5, growth_time=164), label='theoretical_growth_sfs')
# plt.plot(lnfreqs, sfs_growth_2(lnfreqs, t=1.02e5, mu=2.5e-4, n_init=5e4, t_bottleneck=0, growth_time=16400), linewidth=5, alpha=0.3, color='r', label='theoretical_slow_growth_sfs')
# plt.plot(lnfreqs, np.flip(sfs_growth_2(lnfreqs, t=1.2e4, mu=1e-5, n_init=5e4, t_bottleneck=1e4, growth_time=164)), linewidth=5, alpha=0.5, color='r', label='theoretical_growth_sfs')

plt.xlabel('log(frequency)')
plt.xticks(np.arange(-13, 1))
# plt.xlim(-11, 0)
# plt.yticks(np.arange(5, 15))
plt.ylabel('N')
plt.legend()
