"""
Created on 27/03/2019 17:27

@author: R Carthigesan

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 18})


def pop_func(t, t_b_1, t_b_2, t_g_1, t_g_2, n_0):
    if t < t_b_1:
        return n_0
    elif t < t_b_2:
        return n_0 * np.exp((t - t_b_1) / t_g_1)
    else:
        return n_0 * np.exp((t_b_2 - t_b_1) / t_g_1) * np.exp((t - t_b_2) / t_g_2)


fig = plt.figure()
ax = fig.add_subplot(111)

bottleneck = int(2.5e6 - 1e4 * np.log(1e10/4e3))

tot_times = np.arange(start=0, stop=141646, step=1)
tot_pop = [pop_func(t=t, t_b_1=0.0, t_b_2=141600.0, t_g_1=15745.626340, t_g_2=13.524649, n_0=6214) for t in tot_times]

plot_start = -240
ax.semilogy(tot_times[plot_start:], tot_pop[plot_start:])
# ax.plot([tot_times[bottleneck] for i in range(1000)], np.linspace(0, 1e10, 1000), linestyle="dashed", color='k',
#         label='147 millenia ago')
ax.fill_between(tot_times[plot_start:], np.zeros(len(tot_times[plot_start:])), tot_pop[plot_start:], alpha=0.5)

xtick_locs = np.array([141400., 141450., 141500., 141550., 141600., 141650., 141700.]) - 54
xtick_labels = [str((141646-i)*(25/1000)) for i in xtick_locs]
plt.xticks(xtick_locs, xtick_labels)
# plt.xticks(xtick_locs, ['' for i in xtick_labels])
# plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# plt.tick_params(axis='y', which='both', left=False, top=False, labelleft=False)
ax.set_xlabel('Millennia ago')
ax.set_ylabel('Population')
ax.set_xlim(141450, 141646)
# plt.legend(framealpha=1.0, facecolor="whitesmoke")
# ax.set_title('Estimated demographic history', pad=20)
