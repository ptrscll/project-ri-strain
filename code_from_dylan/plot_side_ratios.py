# Code to plot side ratios for each model at each timestep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = r'predef_results/100km/strain_sides_time/noninitial_plots/side_data.txt'

data_file = open(data_path, 'r')

total_ratios = 5
L_R = []
R_L = []
L_S = []
R_S = []
LR_S = []
model_num = '0'

fig, axs = plt.subplots(total_ratios, 1, dpi=300, figsize=(8, 24))

def plot_ratios(ratio_array, model_num):
    for i in range(0, total_ratios):
        axs[i].plot(np.arange(1, len(ratio_array[i]) + 1), ratio_array[i], label='Model ' + model_num)


for line in data_file:
    entries = line.split()
    if entries[0] == 'Model':
        if len(L_R) > 0:
            plot_ratios([L_R, R_L, L_S, R_S, LR_S], model_num)
            L_R = []
            R_L = []
            L_S = []
            R_S = []
            LR_S = []
        model_num = entries[1]

    elif entries[0] != '0':
        left = float(entries[0])
        right = float(entries[1])
        suture = float(entries[2])
        
        L_R.append(left / right)
        R_L.append(right / left)
        L_S.append(left / suture)
        R_S.append(right / suture)
        LR_S.append((left + right) / suture)


plot_ratios([L_R, R_L, L_S, R_S, LR_S], model_num)
data_file.close()

# Making nicer formatting

for i in range(0, total_ratios):
    axs[i].set_xlabel('Timestep')
    axs[i].legend()

axs[0].set_ylabel('left : right')
axs[1].set_ylabel('right : left')
axs[2].set_ylabel('left : suture')
axs[3].set_ylabel('right : suture')
axs[4].set_ylabel('left + right : suture')

axs[2].set_yscale('log')
axs[3].set_yscale('log')
axs[4].set_yscale('log')
plt.tight_layout()

plt.savefig(r'predef_results/100km/strain_sides_time/noninitial_plots/ratio_plots.png')


            