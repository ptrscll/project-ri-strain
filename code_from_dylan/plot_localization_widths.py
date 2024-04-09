# Code to plot widths of central zones at each time step
# Currently configured to read txt files for max-based cutoffs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = r'predef_results/100km/strain_sides_time/noninitial_plots/diff_data_max_10_cutoff.txt'

data_file = open(data_path, 'r')

width_dict = {'left': [], 'right': [], 'suture': [], 'asth': []}
total_ratios = 4

model_num = '0'

fig, axs = plt.subplots(total_ratios, 1, dpi=300, figsize=(8, 20))

def plot_ratios(width_dict, model_num):
    i = 0
    for key in width_dict:
        axs[i].plot(np.arange(1, len(width_dict[key]) + 1), width_dict[key], label='Model ' + model_num)
        i += 1


for line in data_file:
    entries = line.split()
    if entries[0] == 'Model':
        if len(width_dict['left']) > 0:
            plot_ratios(width_dict, model_num)
            width_dict = {'left': [], 'right': [], 'suture': [], 'asth': []}
        model_num = entries[1]

    elif entries[0] in width_dict:
        key = entries[0]
        width = float(entries[1])
        if (width != 0):
            width_dict[key].append(width)
        


plot_ratios(width_dict, model_num)
data_file.close()

# Making nicer formatting
i = 0
for key in width_dict:
    axs[i].set_xlabel('Timestep')
    axs[i].set_ylabel(key + ' side width')
    axs[i].legend()
    i += 1

plt.tight_layout()

plt.savefig(r'predef_results/100km/strain_sides_time/noninitial_plots/width_plots.png')


            