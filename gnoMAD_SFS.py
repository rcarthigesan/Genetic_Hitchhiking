"""
Created on 07/02/2019 11:21

@author: R Carthigesan

"""

import vcf
import re
import numpy as np
# import matplotlib.pyplot as plt
# import random
# from scipy.stats import linregress
# import sim_methods as sim


consequences = ['missense_variant', 'synonymous_variant', 'nonsense_variant']
# frequencies = np.load('ExAC_frequencies.npy').item()
#
# def abline(slope, intercept, lab, col):
#
#     """Plot a line from slope and intercept"""
#     axes = plt.gca()
#     x_vals = np.array(axes.get_xlim())
#     y_vals = intercept + slope * x_vals
#     plt.plot(x_vals, y_vals, '--', label=lab, color=col)
#
# # Extract frequencies from vcf
#

frequencies = {}

for csq in consequences:
    frequencies[csq] = []
    frequencies['afr_' + csq] = []

vcf_reader = vcf.Reader(open("D:/Ramana/gnomad.exomes.r2.1.sites.vcf", 'r'))

# Extract frequencies

count = 0

for record in vcf_reader:

    if record.FILTER:
        # skip FILTER variants
        continue

    ff = record.INFO['vep']
    t1 = ff[0]
    t2 = t1.replace("|", ",")
    positions_of_columns = [m.start() for m in re.finditer(",", t2)]
    p1 = positions_of_columns[0] + 1
    p2 = positions_of_columns[1]
    func_con = t2[p1:p2]

    allele_count = record.INFO["AC"]
    if len(allele_count):
        ac = allele_count[0]

    allele_number = record.INFO["AN"][0]
    freq = float(ac) / allele_number

    afr_allele_count = record.INFO["AC_afr"]
    if len(afr_allele_count):
        afr_ac = afr_allele_count[0]

    afr_allele_number = record.INFO["AN_afr"][0]
    if afr_allele_number != 0:
        afr_freq = float(afr_ac) / afr_allele_number
    else:
        afr_freq = 'undefined'

    if func_con in consequences:
            frequencies[func_con].append(freq)
            if afr_freq != 'undefined' and afr_freq != 0:
                frequencies['afr_' + func_con].append(afr_freq)

    count += 1

    if count % 10000 == 0:
        print("count=", count)

    if count % 10000000 == 0:
        np.save('gnoMAD.npy', frequencies)
        break


# n_samples = min(map(len, list(frequencies.values())))
#
# n_bins = 30
# plt.figure()
#
# colours = []
#
# for i in consequences:
#     colours.append(sim.generate_new_colour(colours, pastel_factor=0))
#
# count = 0
#
# for csq in consequences:
#     # print('# ' + csq + ": " + str(len(frequencies[csq])))
#     freqs_downsampled = [frequencies[csq][i] for i in sorted(random.sample(range(len(frequencies[csq])), n_samples))]
#     histogram = np.histogram(freqs_downsampled, bins=n_bins)
#     midpoints = ((histogram[1])[1:] + (histogram[1])[:-1]) / 2
#     # plt.loglog(midpoints, histogram[0], marker='x', label=csq[:-8])
#     plt.plot(np.log(midpoints), np.log(histogram[0]), marker='x', label=csq[:-8], color=colours[count])
#     linear_fit = linregress(np.log(midpoints), np.log(histogram[0]))
#     linear_fit_except_first = linregress(np.log(midpoints)[1:], np.log(histogram[0])[1:])
#     abline(linear_fit_except_first.slope, linear_fit_except_first.intercept,
#            "All points except first: r = " + str(np.round(linear_fit_except_first.rvalue, 4)), col=colours[count])
#     count += 1
#
# plt.title('ExAC data')
# plt.legend()
