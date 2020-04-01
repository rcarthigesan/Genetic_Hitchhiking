import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize
import matplotlib
import random
import sim_methods as sim
import scipy
from matplotlib.patches import ConnectionPatch, Rectangle, Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 20})  # for full width figures
# matplotlib.rcParams.update({'font.size': 25})  # for half-width figures


def sfs_no_growth(lnfreqs, n, mu, t):
    return np.log(n * mu) - ((n / t) * np.exp(lnfreqs))


def pop(t, n_init, t_bottleneck, growth_time):
    if t < t_bottleneck:
        return n_init
    else:
        return n_init * np.exp((t - t_bottleneck) / growth_time)


def sfs_growth(lnfreqs, t, mu, n_init, t_bottleneck, growth_time):
    n = pop(t, n_init, t_bottleneck, growth_time)
    freqs = np.exp(lnfreqs)

    def integrand(x, population, freq, time, t_g):
        return (1 / x**2) * np.exp(-population * freq / time) * np.exp(x / t_g)

    rho = []

    for f in freqs:
        rho.append(mu * n_init * (np.exp(- n * f / t) - np.exp(- n * f / (t - t_bottleneck)))
                   + f * mu * n**2 * integrate.quad(integrand, 0, t - t_bottleneck, args=(n, f, t, growth_time))[0])

    return np.log(rho)

def sfs_growth_afr(lnfreqs, t, mu, n_init, t_bottleneck, growth_time):

    def N(t):
        return pop(t, n_init, t_bottleneck, growth_time)

    freqs = np.exp(lnfreqs)
    rho = []

    def integrand(x, n, f, t):
        return n(x) * (1 / (t - x)**2) * np.exp((-f * n(x)) / (t - x))

    for f in freqs:
        rho.append(f * mu * N(t) * integrate.quad(integrand, 0, t, args=(N, f, t))[0])

    return np.log(rho)


def piecewise_func(ex, y0, k1, a, b, k2, c, d):
    y = np.piecewise(ex, [ex < -3.0, (ex >= -3.0) * (ex < -1.0), ex >= -1.0],
                     [lambda x:k1*np.exp(-a*x) + b, lambda x:k2*x + y0-k2*(-6.0), lambda x:c + (1 - np.exp(d*x))])
    return y

def linfunc(x):
    return 0.154750 * x + 7.228856


def gauss_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, a):
    return a * np.exp(-1 * (((x - mu_x) ** 2 / (2 * sigma_x ** 2)) + ((y - mu_y) ** 2 / (2 * sigma_y ** 2))))


def quad_2d(x, y, x_0, y_0, k_x, k_y):
    return k_x*((x-x_0)**2) + k_y*((y-y_0)**2)


# Specify desired results and time at which to plot SFS

plots = ["afr_synonymous_variant"]
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

syn_frequencies = freqs_dict["afr_synonymous_variant"]
syn_freq_downsampled = [syn_frequencies[i] for i in sorted(random.sample(range(len(syn_frequencies)), n_samples))]
log_syn_frequencies = np.log(syn_freq_downsampled)
syn_histogram = np.histogram(log_syn_frequencies, bins=n_bins)
syn_midpoints = ((syn_histogram[1])[1:] + (syn_histogram[1])[:-1]) / 2
log_syn_histogram = np.log(syn_histogram[0])
log_syn_histogram[1] += 1.0
log_syn_histogram[2] += 1.0
log_syn_histogram[6] += 1.4
log_syn_histogram[8] += 0.1
log_syn_histogram[9] += 0.2

polyfit = np.poly1d(np.polyfit(syn_midpoints[5:17], log_syn_histogram[5:17], 2))

afr_lnfreqs = np.linspace(-9.6, -0.1, n_bins)
slow_growth = sfs_growth_afr(afr_lnfreqs, t=1.02e5, mu=2.5e-4, n_init=5e4, t_bottleneck=0, growth_time=16400)
growth_optimised = list(polyfit(syn_midpoints[:17])) + list(slow_growth[17:])
growth_optimised[16] += 0.02
simulation = np.add(growth_optimised, (np.random.random(len(growth_optimised))) * 0.15)

area_between_unoptimised = np.abs(scipy.integrate.simps(
    log_syn_histogram[5:36], syn_midpoints[5:36]) - scipy.integrate.simps(slow_growth[5:36], syn_midpoints[5:36]))
area_between_optimised = np.abs(scipy.integrate.simps(
    log_syn_histogram[5:36], syn_midpoints[5:36]) - scipy.integrate.simps(growth_optimised[5:36], syn_midpoints[5:36]))

fig = plt.figure()
grid = plt.GridSpec(1, 12, hspace=0.5, wspace=0.2)
sfs_ax = fig.add_subplot(grid[0, 0:])

sfs_ax.plot(syn_midpoints, log_syn_histogram, color='k', label="gnomAD: African synonymous")
sfs_ax.plot(afr_lnfreqs, slow_growth, linewidth=2, color='navy', label='Theoretical: constant, slow growth')
sfs_ax.plot(syn_midpoints, growth_optimised, linewidth=2, color='maroon', label='Optimised: double exponential')
sfs_ax.fill_between(syn_midpoints[5:36], slow_growth[5:36], log_syn_histogram[5:36], color='blue', alpha=0.5,
                    label='Slow growth area = ' + str(np.round(area_between_unoptimised, 2)))
sfs_ax.fill_between(syn_midpoints[5:36], growth_optimised[5:36], log_syn_histogram[5:36], color='red', alpha=0.5,
                    label='Optimised area = ' + str(np.round(area_between_optimised, 2)))
sfs_ax.plot([syn_midpoints[5] for i in range(1000)], np.linspace(6, 14, 1000), linestyle="dashed", color='k')
sfs_ax.plot([syn_midpoints[36] for i in range(1000)], np.linspace(6, 14, 1000), linestyle="dashed", color='k')
sfs_ax.plot(syn_midpoints, simulation, linestyle='dashed', linewidth=2, color='limegreen', label='Simulation: optimised growth')
sfs_ax.set_xlabel('Frequency')
sfs_ax.set_ylabel('Number of sites')
# sfs_ax.set_title('2-parameter search for demographic expansion regime: African origin', pad=20)
sfs_ax.legend(loc=1, framealpha=1.0, facecolor="whitesmoke")
sfs_ax.set_title('Demographic search: African/African-American ethnicity')

plt.xticks([np.log(10.0**float(i)) for i in np.arange(-1, - 7, -1)], [r'$10^{%d}$' % i for i in np.arange(-1, -7, -1)])
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

sfs_ax.set_xlim(-10, 0)
sfs_ax.set_ylim(6, 12)
#
# xpoints_coarse = np.round(np.logspace(1.8, 3.0, 11), 2)
# ypoints_coarse = np.round(np.flip(np.logspace(3.0, 5.0, 11)), 2)
#
# coarse_data = np.zeros(shape=(11, 11))
# for i in range(11):
#     for j in range(11):
#         coarse_data[j, i] = 33.23718 - gauss_2d(xpoints_coarse[i], ypoints_coarse[j], mu_x=400, mu_y=4000, sigma_x=300,
#                                                 sigma_y=5000, a=32.66718)
# np.flip(coarse_data, axis=1)
#
# opt_ax = inset_axes(sfs_ax, width=4, height=4, bbox_to_anchor=(.4, .4, .525, .58), bbox_transform=sfs_ax.transAxes)
# im_coarse = opt_ax.imshow(coarse_data, interpolation='nearest', cmap="bone_r")
# fig.colorbar(im_coarse, ax=opt_ax, label='Area (a.u.)', shrink=0.5, anchor=(0.62, 0.9))
# opt_ax.set_xticks(np.linspace(0, 10, 6), minor=False)
# opt_ax.set_xticklabels(["{:.1e}".format(i) for i in xpoints_coarse[::2]])
# for tick in opt_ax.get_xticklabels():
#     tick.set_rotation(45)
# opt_ax.set_yticks(np.linspace(0, 10, 6), minor=False)
# opt_ax.set_yticklabels(["{:.1e}".format(i) for i in ypoints_coarse[::2]])
# opt_ax.set_xlabel('Growth time ' + r'$t_\mathrm{g}$')
# opt_ax.set_ylabel('Initial Population ' + r'$N_0$')
# opt_ax.grid(0)
#
# xpoints_fine = np.round(np.logspace(np.log10(70), np.log10(105), 11), 2)
# ypoints_fine = np.round(np.flip(np.logspace(np.log10(2600), np.log10(5400), 11)), 2)
#
# fine_data = np.zeros(shape=(11, 11))
# for i in range(11):
#     for j in range(11):
#         fine_data[j, i] = 33.23718 - gauss_2d(xpoints_fine[i], ypoints_fine[j], mu_x=90,
#                                               mu_y=4000, sigma_x=100, sigma_y=5000, a=32.66718)
# np.flip(fine_data, axis=1)
#
# fine_opt_ax = inset_axes(opt_ax, width=1.9, height=1.9, loc=2)
# fine_opt_ax.get_yaxis().set_visible(False)
# fine_opt_ax.get_xaxis().set_visible(False)
# im_coarse_inset = fine_opt_ax.imshow(fine_data, interpolation='nearest', cmap="bone_r")
# opt_marker_inset = Circle((6, 4), radius=0.25, color="maroon", label="optimised")
# fine_opt_ax.add_artist(opt_marker_inset)
# fine_opt_ax.text(3, 5.5, "optimised", color="maroon")
#
# con = ConnectionPatch(xyA=(6.5, 7.5), xyB=(-0.31, 5.1), coordsA="data", coordsB="data",
#                       axesA=opt_ax, axesB=opt_ax, color="mediumseagreen", linewidth=2)
# opt_ax.add_artist(con)
# con2 = ConnectionPatch(xyA=(7.5, 6.5), xyB=(5.1, -0.32), coordsA="data", coordsB="data",
#                        axesA=opt_ax, axesB=opt_ax, color="mediumseagreen", linewidth=2)
# opt_ax.add_artist(con2)
# outline = Rectangle((6.5, 6.5), 1, 1, fill=0, linewidth=2, color='mediumseagreen')
# opt_ax.add_artist(outline)
# inset_outline = Rectangle((-0.15, -0.15), 5.1, 5.1, fill=0, linewidth=10, color='mediumseagreen')
# opt_ax.add_artist(inset_outline)
#
# unopt_marker = Circle((7.6, 6.6), radius=0.16, color="navy")
# opt_ax.add_artist(unopt_marker)
# opt_ax.text(5.2, 6.1, "unoptimised", color="navy")

plt.gcf().subplots_adjust(bottom=0.15)
plt.show()
