# Code to plot suture widths vs. side ratios for each model at each timestep
# Ultimate plot should resemble a P-T-t plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


side_path = r'predef_results/100km/strain_sides_time/noninitial_plots/side_data.txt'
width_path = r'predef_results/100km/strain_sides_time/noninitial_plots/diff_data_max_10_cutoff.txt'


# Getting side ratios
data_file = open(side_path, 'r')

R_L = []

fig, ax = plt.subplots(dpi=300)

for line in data_file:
    entries = line.split()
    if entries[0] == 'Model':
        R_L.append([])

    elif entries[0] != '0':
        left = float(entries[0])
        right = float(entries[1])
        suture = float(entries[2])
        
        R_L[-1].append(right / left)

data_file.close()


# Getting suture widths
data_file = open(width_path, 'r')

widths = []

for line in data_file:
    entries = line.split()
    if entries[0] == 'Model':
        widths.append([])
        
    elif entries[0] == 'suture':
        width = float(entries[1])
        if (width != 0):
            widths[-1].append(width)
        
data_file.close()

# Plotting data
for i in range(0, 8):
    ax.plot(R_L[i], widths[i], label='Model ' + str(i), marker='X', markevery=[-1])


# Making nicer formatting
ax.set_xlabel('Right Side Strain : Left Side Strain')
ax.set_ylabel('Suture Width')

ax.legend()
ax.set_xscale('log')
plt.tight_layout()

plt.savefig(r'predef_results/100km/strain_sides_time/noninitial_plots/L_R_S_W_plot.png')


            
