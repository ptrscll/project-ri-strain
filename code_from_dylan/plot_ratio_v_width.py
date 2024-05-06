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
def add_arrow(line, position=None, direction='right', size=15, color=None, start_ind = None, alpha = 1.0):
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
        arrowprops=dict(arrowstyle="->", color=color, alpha=alpha),
        size=size, alpha=alpha
    )


# Getting side ratios
data_file = open(side_path, 'r')

R_L = []

fig, ax = plt.subplots(dpi=300, figsize=(10, 9))

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(16)
plt.rcParams.update({'font.size': 16})
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
#ax.get_xticklabels().set_fontsize(16)

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
    if i + 1 not in [1, 3, 5, 8]:
        alpha = 0.4
        line.set_color('gray')
    else:
        alpha = 1.0
        if i == 7:
            line.set_color('red')
        plt.scatter(R_L[i][:-1], widths[i][:-1], 20, color=line.get_color())
    line.set_alpha(alpha)
    if i + 1 not in [1, 3, 5, 8]:
        add_arrow(line, start_ind=1, alpha=alpha)
        add_arrow(line, start_ind=4, alpha=alpha)
        add_arrow(line, start_ind=7, alpha=alpha)
    elif i + 1 == 1:
        add_arrow(line, start_ind=0, alpha=alpha)
        add_arrow(line, start_ind=3, alpha=alpha)
        add_arrow(line, start_ind=6, alpha=alpha)
    elif i + 1 == 3:
        add_arrow(line, start_ind=1, alpha=alpha)
        add_arrow(line, start_ind=2, alpha=alpha)
        add_arrow(line, start_ind=5, alpha=alpha)
    elif i + 1 == 5:
        add_arrow(line, start_ind=1, alpha=alpha)
        add_arrow(line, start_ind=6, alpha=alpha)
    elif i + 1 == 8:
        add_arrow(line, start_ind=0, alpha=alpha)
        add_arrow(line, start_ind=3, alpha=alpha)
        add_arrow(line, start_ind=5, alpha=alpha)



# Making nicer formatting
ax.set_xscale('log')
ax.set_xlabel('Right Side Strain : Left Side Strain')
ax.set_ylabel('Width of High Strain Zone (km)')
ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
ax.set_xticklabels([r'$\frac{1}{4}$', r'$\frac{1}{2}$', '1', '2', '4'])
ax.set_xticks([2**(-1.5), 2**(-0.5), 2**(0.5), 2**(1.5)], minor=True, labels=[])

# Centering x-axis at x=1
x_min, x_max = ax.get_xlim()
new_max = max(1.0 / x_min, x_max)
ax.set_xlim(1.0 / new_max, new_max)

# Adjusting y-axis
ax.set_ylim(0, ax.get_ylim()[1])

# Adding symmetry shading and width boundary line
ax.axvspan(2**(-0.5), 2**(0.5), facecolor='orange', alpha=0.25)
ax.axhline(175, color='black', linestyle='dashed')

ax.legend()

# Circling timesteps
for i in [0, 2, 4, 7]:
    tsteps = []
    if i == 0 or i == 4:
        tsteps = np.array([5, 10])
    else:
        tsteps = np.array([3, 10])
    tsteps -= 1
    for j in tsteps:
        ax.plot(R_L[i][j], widths[i][j], 'o', ms=10, mec='black', mfc='none', mew=1)

plt.tight_layout()

plt.savefig(r'predef_results/100km/strain_sides_time/noninitial_plots/full_width_ratio_plot.png')


            
