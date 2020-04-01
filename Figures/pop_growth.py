"""
Created on 27/03/2019 17:27

@author: R Carthigesan

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 18})


def pop_func(t, t_b, t_g, n_0):
    if t < t_b:
        return n_0
    else:
        return n_0 * np.exp((t - t_b) / t_g)


fig = plt.figure()
ax = fig.add_subplot(111)

bottleneck = int(2.5e6 - 1e4 * np.log(1e10/4e3))

tot_times = np.arange(start=0, stop=2.5e6, step=1)
tot_pop = [pop_func(t=t, t_b=bottleneck, t_g=1e4, n_0=4e3) for t in tot_times]

plot_start = int(2.5e6 - 2*(2.5e6 - bottleneck))
ax.semilogy(tot_times[plot_start:], tot_pop[plot_start:])
ax.plot([tot_times[bottleneck] for i in range(1000)], np.linspace(0, 1e10, 1000), linestyle="dashed", color='k',
        label='144 millennia ago')
ax.fill_between(tot_times[plot_start:], np.zeros(len(tot_times[plot_start:])), tot_pop[plot_start:], alpha=0.5)

xtick_locs = np.linspace(2.2e6, 2.5e6, 7)
xtick_labels = [str(int((2.5e6 - i)/1000)) for i in xtick_locs]
plt.xticks(xtick_locs, xtick_labels)
# plt.xticks(xtick_locs, ['' for i in xtick_labels])
# plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# plt.tick_params(axis='y', which='both', left=False, top=False, labelleft=False)
ax.set_xlabel('Millennia ago')
ax.set_ylabel('Population')
ax.set_xlim(2.21e6, 2.51e6)
plt.legend(framealpha=1.0, facecolor="whitesmoke")
# ax.set_title('Estimated demographic history', pad=20)
