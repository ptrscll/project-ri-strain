# Code to plot side strains for each model at each timestep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = r'predef_results/100km/strain_sides_time/noninitial_plots/side_data.txt'

data_file = open(data_path, 'r')

side_dict = {'left': [], 'right': [], 'suture': [], 'asth': []}
total_sides = 4

model_num = '0'

fig, axs = plt.subplots(total_sides, 1, dpi=300, figsize=(8, 20))

def plot_sides(side_dict, model_num):
    i = 0
    for key in side_dict:
        axs[i].plot(np.arange(1, len(side_dict[key]) + 1), side_dict[key], label='Model ' + model_num)
        i += 1


for line in data_file:
    entries = line.split()
    if entries[0] == 'Model':
        if len(side_dict['left']) > 0:
            plot_sides(side_dict, model_num)
            side_dict = {'left': [], 'right': [], 'suture': [], 'asth': []}
        model_num = entries[1]

    elif entries[0] != '0':
        side_dict['left'].append(float(entries[0]))
        side_dict['right'].append(float(entries[1]))
        side_dict['suture'].append(float(entries[2]))
        side_dict['asth'].append(float(entries[3]))
   


plot_sides(side_dict, model_num)
data_file.close()

# Making nicer formatting
i = 0
for key in side_dict:
    axs[i].set_xlabel('Timestep')
    axs[i].set_ylabel(key + ' total strain')
    axs[i].legend()
    i += 1

plt.tight_layout()

plt.savefig(r'predef_results/100km/strain_sides_time/noninitial_plots/side_plots.png')


            
