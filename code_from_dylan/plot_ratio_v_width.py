# Code to plot suture widths vs. side ratios for each model at each timestep
# Ultimate plot should resemble a P-T-t plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


side_path = r'predef_results/100km/strain_sides_time/noninitial_plots/side_data.txt'
width_path = r'predef_results/100km/strain_sides_time/noninitial_plots/width_data.txt'

# Code to add arrows to plots
# Adapted from https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot
def add_arrow(line, position=None, direction='right', size=15, color=None, start_ind = None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    if start_ind == None:
        start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


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
        
    #elif entries[0] == 'suture':
    else:
        width = float(entries[0])
        if (width != 0):
            widths[-1].append(width)
        
data_file.close()

# Plotting data
for i in range(0, 8):
    line = ax.plot(R_L[i], widths[i], label='Model ' + str(i + 1), marker='X', markevery=[-1])[0]
    add_arrow(line, start_ind=1)
    add_arrow(line, start_ind=4)
    add_arrow(line, start_ind=7)


# Making nicer formatting
ax.set_xlabel('Right Side Strain : Left Side Strain')
ax.set_ylabel('Width of Primary Strain Regions (km)')
plt.tick_params(axis='x', which='minor')
ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

ax.legend()
ax.set_xscale('log')
plt.tight_layout()

plt.savefig(r'predef_results/100km/strain_sides_time/noninitial_plots/full_width_ratio_plot.png')


            
