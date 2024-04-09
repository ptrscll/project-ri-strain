# Code to plot width ratios at each time step
# Currently configured to read txt files for max-based cutoffs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = r'predef_results/100km/strain_sides_time/noninitial_plots/diff_data_max_10_cutoff.txt'

data_file = open(data_path, 'r')

width_dict = {'left': [], 'right': [], 'suture': [], 'asth': []}
total_ratios = 4

model_num = '0'

fig, axs = plt.subplots(8, 1, dpi=300, figsize=(6, 30))

def plot_ratios(width_dict, model_num):
    curr_label = 'Model ' + model_num
    left = np.array(width_dict['left'])
    right = np.array(width_dict['right'])
    suture = np.array(width_dict['suture'])
    tsteps = np.arange(1, len(left) + 1)
    axs[0].plot(tsteps, left /right, label=curr_label)
    axs[1].plot(tsteps, right / left, label=curr_label)
    axs[2].plot(tsteps, left / suture, label=curr_label)
    axs[3].plot(tsteps, right / suture, label=curr_label)
    axs[4].plot(tsteps, (left + right) / suture, label=curr_label) 
    axs[5].plot(tsteps, suture / left, label=curr_label)
    axs[6].plot(tsteps, suture / right, label=curr_label)
    axs[7].plot(tsteps, suture / (left + right), label = curr_label)

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
for i in range(0, 8):
    axs[i].set_xlabel('Timestep')
    axs[i].legend()

axs[0].set_ylabel('left : right')
axs[1].set_ylabel('right : left')
axs[2].set_ylabel('left : suture')
axs[3].set_ylabel('right : suture')
axs[4].set_ylabel('left + right : suture')
axs[5].set_ylabel('suture : left')
axs[6].set_ylabel('suture : right')
axs[7].set_ylabel('suture : left + right')

plt.tight_layout()

plt.savefig(r'predef_results/100km/strain_sides_time/noninitial_plots/width_ratio_plots.png')


            
