"""
Created on Sun 27 Jan 13:24:34 2019

@author: R Carthigesan

"""

# Import libraries and define useful plotting function

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize
import matplotlib
import random
import sim_methods as sim

plt.style.use('ggplot')
# matplotlib.rcParams.update({'font.size': 20})  # for full width figures
matplotlib.rcParams.update({'font.size': 45})  # for half-width figures


def sfs_no_growth(lnfreqs, N, mu, t):
    return np.log(N*mu) - ((N / t) * np.exp(lnfreqs))


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
        rho.append(mu * n_init * (np.exp(- N * f / t) - np.exp(- N * f / (t - t_bottleneck)))
            + f * mu * N**2 * integrate.quad(integrand, 0, t - t_bottleneck, args=(N, f, t, growth_time))[0])

    return np.log(rho)


def piecewise_func(x, y0, k1, a, b, k2, c, d):
    y = np.piecewise(x, [x < -3.0, (x >= -3.0)*(x < -1.0), x >= -1.0],
                     [lambda x:k1*np.exp(-a*x) + b, lambda x:k2*x + y0-k2*(-6.0), lambda x:c + (1 - np.exp(d*x))])
    return y


def linfunc(x):
    return 0.154750 * x + 7.228856


# Specify desired results and time at which to plot SFS

plots = ["synonymous_variant"]
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

syn_frequencies = freqs_dict["synonymous_variant"]
syn_freq_downsampled = [syn_frequencies[i] for i in sorted(random.sample(range(len(syn_frequencies)), n_samples))]
log_syn_frequencies = np.log(syn_freq_downsampled)
syn_histogram = np.histogram(log_syn_frequencies, bins=n_bins)
syn_midpoints = ((syn_histogram[1])[1:] + (syn_histogram[1])[:-1]) / 2
log_syn_histogram = np.log(syn_histogram[0])
for i in np.arange(-13, 0):
    log_syn_histogram[i] -= (linfunc(syn_midpoints[i]) - 6.74755)
polyfit = np.poly1d(np.polyfit(syn_midpoints, np.log(syn_histogram[0]), 20))
curvefit = optimize.curve_fit(piecewise_func, syn_midpoints[4:], np.log(syn_histogram[0])[4:])[0]

fit = piecewise_func(syn_midpoints[4:], *curvefit)
addon = piecewise_func(syn_midpoints[:4], *curvefit)
no_growth = sfs_no_growth(syn_midpoints, N=1e10, mu=1.2e-7, t=1e10)
cross_point = 33
shift = 6
cutoff = -6
growth = list(piecewise_func(syn_midpoints[:4], *curvefit))\
         + list(fit[:cross_point]) + list(no_growth[4+cross_point-shift:-shift] - 0.36)
growth[-13] += 0.02
growth[-14] += 0.01
log_syn_histogram[35] += 0.05

linear_adjust = list(np.flip(np.linspace(0.0001, 0.6, 37) * -1)) + list(np.zeros(len(growth) - 37))
growth = np.add(np.array(growth), 0.3 * np.array(linear_adjust))
fast_growth = np.add(np.array(growth), (-2) * np.array(linear_adjust))
slow_growth = np.add(np.array(growth), 2 * np.array(linear_adjust))

fig = plt.figure(figsize=(3.2 * 5.5,1.68 * 5.5))
ax = fig.add_subplot(111)

ax.plot(syn_midpoints, fast_growth, linewidth=6, color='orange', label='Fast growth rate')
ax.plot(syn_midpoints, growth, linewidth=6, color='red', label='Medium growth rate')
ax.plot(syn_midpoints, slow_growth, linewidth=6, color='darkred', label='Slow growth rate')


plt.xticks([np.log(10.0**float(i)) for i in np.arange(-1,-7,-1)], [r'$10^{%d}$' % (i) for i in np.arange(-1,-7,-1)])
plt.yticks([np.log(10.0**float(i)) for i in np.arange(2,7,1)], [r'$10^{%d}$' % (i) for i in np.arange(2,7,1)])

minor_ticks_x = []
for i in np.arange(-1.0,-7.0,-1.0):
    for j in np.arange(1.0,10.0,1.0):
        minor_ticks_x.append(j * (10 ** i))
ax.set_xticks([np.log(i) for i in minor_ticks_x], minor=True)

minor_ticks_y = []
for i in np.arange(2.0,8.0,1.0):
    for j in np.arange(1.0,10.0,1.0):
        minor_ticks_y.append(j * (10 ** i))
ax.set_yticks([np.log(i) for i in minor_ticks_y], minor=True)

plt.ylim(6, 15)
plt.xlim(-13, 0)
plt.xlabel('Frequency')
plt.ylabel('Number of sites')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
# plt.title('gnomAD SFS: synonymous', pad=10)
plt.tight_layout()
plt.legend(framealpha=1.0, facecolor="whitesmoke")
