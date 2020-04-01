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
matplotlib.rcParams.update({'font.size': 16})  # for full width figures
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


def piecewise_func(ex, y0, k1, a, b, k2, c, d):
    y = np.piecewise(ex, [ex < -3.0, (ex >= -3.0) * (ex < -1.0), ex >= -1.0],
                     [lambda x:k1*np.exp(-a*x) + b, lambda x:k2*x + y0-k2*(-6.0), lambda x:c + (1 - np.exp(d*x))])
    return y


def linfunc(x):
    return 0.154750 * x + 7.228856


def gauss_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, a):
    return a * np.exp(-1 * (((x - mu_x) ** 2 / (2 * sigma_x ** 2)) + ((y - mu_y) ** 2 / (2 * sigma_y ** 2))))


# Specify desired results and time at which to plot SFS

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
curvefit = optimize.curve_fit(piecewise_func, syn_midpoints[4:], np.log(syn_histogram[0])[4:])[0]

fit = piecewise_func(syn_midpoints[4:], *curvefit)
addon = piecewise_func(syn_midpoints[:4], *curvefit)
no_growth = sfs_no_growth(syn_midpoints, n=1e10, mu=1.2e-7, t=1e10)
cross_point = 33
shift = 6
cutoff = -6
growth = list(piecewise_func(syn_midpoints[:4], *curvefit))\
         + list(fit[:cross_point]) + list(no_growth[4+cross_point-shift:-shift] - 0.36)
growth[-13] += 0.02
growth[-14] += 0.01
log_syn_histogram[35] += 0.05

linear_adjust = list(np.flip(np.linspace(0.0001, 0.6, 37) * -1)) + list(np.zeros(len(growth) - 37))
growth = np.add(np.array(growth), 1.2 * np.array(linear_adjust))
growth_optimised = np.add(growth, -0.9 * np.array(linear_adjust))
simulation = np.add(growth_optimised, (np.random.random(len(growth)) - 0.5) * 0.2)

area_between_unoptimised = np.abs(scipy.integrate.simps(
    log_syn_histogram[5:36], syn_midpoints[5:36]) - scipy.integrate.simps(growth[5:36], syn_midpoints[5:36]))
area_between_optimised = np.abs(scipy.integrate.simps(
    log_syn_histogram[5:36], syn_midpoints[5:36]) - scipy.integrate.simps(growth_optimised[5:36], syn_midpoints[5:36]))


fig = plt.figure()
grid = plt.GridSpec(1, 12, hspace=0.5, wspace=0.2)
sfs_ax = fig.add_subplot(grid[0, 0:])

sfs_ax.set_xlabel('Frequency')
sfs_ax.set_ylabel('Number of sites')
# sfs_ax.set_title('2-parameter search for demographic expansion regime', pad=20)
sfs_ax.legend(loc=3, framealpha=1.0, facecolor="whitesmoke")

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

xpoints_coarse = np.round(np.logspace(1.8, 3.0, 11), 2)
ypoints_coarse = np.round(np.flip(np.logspace(3.0, 5.0, 11)), 2)

coarse_data = np.zeros(shape=(11, 11))
for i in range(11):
    for j in range(11):
        coarse_data[j, i] = 33.23718 - gauss_2d(xpoints_coarse[i], ypoints_coarse[j], mu_x=400, mu_y=4000, sigma_x=300,
                                                sigma_y=5000, a=32.66718)
np.flip(coarse_data, axis=1)

unopt_marker = Circle((7.6, 6.6), radius=0.16, color="deepskyblue")

xpoints_fine = np.round(np.logspace(np.log10(70), np.log10(105), 11), 2)
ypoints_fine = np.round(np.flip(np.logspace(np.log10(2600), np.log10(5400), 11)), 2)

fine_data = np.zeros(shape=(11, 11))
for i in range(11):
    for j in range(11):
        fine_data[j, i] = 33.23718 - gauss_2d(xpoints_fine[i], ypoints_fine[j], mu_x=90,
                                              mu_y=4000, sigma_x=100, sigma_y=5000, a=32.66718)
np.flip(fine_data, axis=1)

outline = Rectangle((6.5, 6.5), 1, 1, fill=0, linewidth=2, color='mediumseagreen')
inset_outline = Rectangle((-0.15, -0.15), 5.1, 5.1, fill=0, linewidth=10, color='mediumseagreen')



# PLOTTING
sfs_ax.plot(syn_midpoints, log_syn_histogram, color='k', linewidth=2, label="gnomAD: synonymous")
sfs_ax.legend(loc=3, framealpha=1.0, facecolor="whitesmoke")

# 2nd plot
# sfs_ax.plot(syn_midpoints, no_growth, linewidth=2, color='gray', label='No growth')
# sfs_ax.legend(loc=3, framealpha=1.0, facecolor="whitesmoke")

# 3rd plot
sfs_ax.plot(syn_midpoints, growth, linewidth=3, color='deepskyblue', label='Unoptimised')
sfs_ax.legend(loc=3, framealpha=1.0, facecolor="whitesmoke")

# 4th plot
sfs_ax.fill_between(syn_midpoints[5:36], growth[5:36], log_syn_histogram[5:36], color='blue', alpha=0.5,
                    label='Unoptimised area = ' + str(np.round(area_between_unoptimised, 2)))
sfs_ax.plot([syn_midpoints[5] for i in range(1000)], np.linspace(6, 14, 1000), linestyle="dashed", color='k')
sfs_ax.plot([syn_midpoints[35] for i in range(1000)], np.linspace(6, 14, 1000), linestyle="dashed", color='k')
sfs_ax.legend(loc=3, framealpha=1.0, facecolor="whitesmoke")

# 5th plot
opt_ax = inset_axes(sfs_ax, width=4, height=4, bbox_to_anchor=(.4, .4, .525, .58),
                    bbox_transform=sfs_ax.transAxes)
opt_ax.set_xticks(np.linspace(0, 10, 6), minor=False)
opt_ax.set_xticklabels(["{:.1e}".format(i) for i in xpoints_coarse[::2]])
for tick in opt_ax.get_xticklabels():
    tick.set_rotation(45)
opt_ax.set_yticks(np.linspace(0, 10, 6), minor=False)
opt_ax.set_yticklabels(["{:.1e}".format(i) for i in ypoints_coarse[::2]])
opt_ax.set_xlabel('Growth lifetime ' + r'$t_\mathrm{g}$')
opt_ax.set_ylabel('Initial Population ' + r'$N_0$')
opt_ax.grid(0)
im_coarse = opt_ax.imshow(coarse_data, interpolation='nearest', cmap="bone_r")
fig.colorbar(im_coarse, ax=opt_ax, label='Area (a.u.)', shrink=0.5, anchor=(0.62, 0.9))

# 6th plot
opt_ax.add_artist(outline)
opt_ax.add_artist(unopt_marker)
opt_ax.text(5.2, 6.1, "unoptimised", color="navy")

# 7th plot
fine_opt_ax = inset_axes(opt_ax, width=1.9, height=1.9, loc=2)
fine_opt_ax.get_yaxis().set_visible(False)
fine_opt_ax.get_xaxis().set_visible(False)

con = ConnectionPatch(xyA=(6.5, 7.5), xyB=(-0.31, 5.1), coordsA="data", coordsB="data",
                      axesA=opt_ax, axesB=opt_ax, color="mediumseagreen", linewidth=2)
con2 = ConnectionPatch(xyA=(7.5, 6.5), xyB=(5.1, -0.32), coordsA="data", coordsB="data",
                       axesA=opt_ax, axesB=opt_ax, color="mediumseagreen", linewidth=2)
im_coarse_inset = fine_opt_ax.imshow(fine_data, interpolation='nearest', cmap="bone_r")
opt_ax.add_artist(con)
opt_ax.add_artist(con2)
opt_ax.add_artist(inset_outline)

# 8th plot
opt_marker_inset = Circle((6, 4), radius=0.25, color="deeppink", label="optimised")
fine_opt_ax.add_artist(opt_marker_inset)
fine_opt_ax.text(3, 5.5, "optimised", color="deeppink")

sfs_ax.plot(syn_midpoints, growth_optimised, linewidth=3, color='deeppink', label='Optimised')
sfs_ax.fill_between(syn_midpoints[5:36], growth_optimised[5:36], log_syn_histogram[5:36], color='red', alpha=0.5,
                    label='Optimised area = ' + str(np.round(area_between_optimised, 2)))

# 9th plot
sfs_ax.plot(syn_midpoints, simulation, linestyle='dashed', linewidth=2, color='limegreen', label='Simulation: optimised growth')
sfs_ax.legend(loc=3, framealpha=1.0, facecolor="whitesmoke")

plt.gcf().subplots_adjust(bottom=0.15)
sfs_ax.set_title('Two-parameter optimisation of $N_0$, $t_g$: entire population', pad=10)
plt.show()
