import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import linregress

matplotlib.rcParams.update({'font.size': 18})

def abline(slope, intercept, lab, col):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', label = lab, color = col)

fourD_sites = np.array([int(line.rstrip('\n')) for line in open(r"C:\Users\raman\OneDrive\Work\Cambridge_MASt_Physics\Project"
                                                 r"\Ramana_PartIII_Project_Shared\Initial_Report\SFS_data"
                                                 r"\4D_sites.txt")])

nonsyn_sites = np. array([int(line.rstrip('\n')) for line in open(r"C:\Users\raman\OneDrive\Work\Cambridge_MASt_Physics\Project"
                                                 r"\Ramana_PartIII_Project_Shared\Initial_Report\SFS_data"
                                                 r"\nonsyn_sites.txt")])

freq = np.array([(i+1)/len(fourD_sites) for i in range(len(fourD_sites))])
fourD_fit = linregress(np.log(freq), np.log(fourD_sites))
nonsyn_fit = linregress(np.log(freq), np.log(nonsyn_sites))
fourD_fit_except_first = linregress(np.log(freq)[1:], np.log(fourD_sites)[1:])
nonsyn_fit_except_first = linregress(np.log(freq)[1:], np.log(nonsyn_sites)[1:])

plt.plot(np.log(freq), np.log(fourD_sites), label = '4D Sites', marker='x', color='royalblue')
abline(fourD_fit.slope, fourD_fit.intercept, "4D linear fit: r = " + str(np.round(fourD_fit.rvalue, 3)), 'royalblue')
abline(fourD_fit_except_first.slope, fourD_fit_except_first.intercept, "4D linear fit except first: r = " + str(np.round(fourD_fit_except_first.rvalue, 3)), 'navy')

plt.plot(np.log(freq), np.log(nonsyn_sites), label = 'Nonsyn Sites', marker='x', color='orange')
abline(nonsyn_fit.slope, nonsyn_fit.intercept, "Nonsyn linear fit: r = " + str(np.round(nonsyn_fit.rvalue, 3)), 'orange')
abline(nonsyn_fit_except_first.slope, nonsyn_fit_except_first.intercept, "Nonsyn linear fit except first: r = " + str(np.round(nonsyn_fit_except_first.rvalue, 3)), 'red')

plt.legend()
plt.xlabel('log(frequency)')
plt.ylabel('log(count)')