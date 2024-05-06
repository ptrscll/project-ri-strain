"""
Script to do strain analysis on either side of the suture for initial rift inversion models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mpltern
from scipy.signal import savgol_filter,find_peaks,peak_widths
from scipy.stats import skew,mode
from tqdm import tqdm

import pyvista as pv
import vtk_plot as vp

# The following 16 models need to be plotted
# slow_cold_half 063022_rip_c
# slow_cold_half_qui 071822_rip_b
# slow_cold_full 070422_rip_e
# slow_cold_full_qui 072022_rip_a
# hot_fast_half 070422_rip_c
# hot_fast_half_qui 071322_rip
# hot_fast_full 070622_rip_a
# hot_fast_full_qui 072022_rip_b

# slow_cold_half 080122_rip_a
# slow_cold_half_qui 080122_rip_e
# slow_cold_full 080122_rip_b
# slow_cold_full_qui 080122_rip_f
# hot_fast_half 080122_rip_c
# hot_fast_half_qui 080122_rip_g
# hot_fast_full 080122_rip_d
# hot_fast_full_qui 080122_rip_h

# Make list of models with appropriate names
models = ['063022_rip_c','071822_rip_b','070422_rip_e','072022_rip_a',
          '070422_rip_c','071322_rip','070622_rip_a','072022_rip_b']
# Fast models (for reference when analyzing them later):
#          '080122_rip_a','080122_rip_e','080122_rip_b','080122_rip_f',
#          '080122_rip_c','080122_rip_g','080122_rip_d','080122_rip_h']

#models = ['063022_rip_c']

names = ['Model ' + str(x) for x in range(1,17)]

output_dir = r'predef_results/100km/strain_sides_time/noninitial_plots/'

# Indicate time of final rift of reach model (post-cooling)

times = [16,36,32,52,7.3,27.3,14.5,34.5]*2
tstep_interval = 0.1
invert_times = [20]*8 + [4]*8

# Set up the figure

bounds = [300,700,400,620]

# Variables for seeing strain distribution
field = 'noninitial_plastic_strain'
opacity_strain = [0, 0.7, 0.7, 0.7, 0.7]
lim_strain = [0, 5]
cm_strain = 'inferno'
subtract_initial = True


# Create dataframe for final values
final = pd.DataFrame([],columns=['Localization','Symmetry','C-ness'])

k = 0
model = models[k]
if k == 0:

    # Setting up variables for plotting loop
    side_dir = r'predef_suture_figs/'

    last_mesh_num = 10
    if model == '071322_rip':
        last_mesh_num = 9
    bounds_mesh = [bound*1e3 for bound in bounds] +[0,0]

    # Setting up figure
    # Currently does NOT create normalized plot
    fig, axs = plt.subplots(3, 2, dpi=300, figsize=(10, 10))

    for i in range(0, 3):
        for ax in axs[i]:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(16)
            plt.rcParams.update({'font.size': 16})
            #ax.get_xticklabels().set_fontsize(16)

    # Setting up dataframe for noninitial strain plotting
    initial_df = pd.DataFrame()
    initial_df['left'] = np.zeros(701)
    initial_df['right'] = np.zeros(701)
    initial_df['suture'] = np.zeros(701)
    initial_df['asth'] = np.zeros(701)

    # Making plots for each timestep
    j = 0
    for i in range(0, last_mesh_num + 1):
        if i in [0, 5, 10]:
            file = side_dir + model + '/' + model + '_' + str(i) + '.vtu'
            mesh = pv.read(file)
            clipped = mesh.clip_box(bounds_mesh)

            # Get all strain values
            x = clipped.points[:,0]
            x_rounded = np.round(x,-3)
            
            # Pull strains, sides, and x positions into dataframe
            strains = clipped[field]
            sides = clipped['rift_side']
            df = pd.DataFrame([x_rounded,strains,sides]).T
            df.columns = ['X','Strain', 'Side&Layer']    
        
            # Splitting the table into left and right sides of the suture
            left_df = df[(df['Side&Layer'] <= 3) & (df['Side&Layer'] != 0)]
            right_df = df[(df['Side&Layer'] > 3) & (df['Side&Layer'] <= 6)]
            suture_df = df[df['Side&Layer'] > 6]
            asth_df = df[df['Side&Layer'] == 0]

            # Setting up plots
            #pv.start_xvfb()
            
            # Plot suture results and strain
            vp.plot2D(file,'rift_side',bounds=bounds,ax=axs[j][0])
            vp.plot2D(file,field,bounds,ax=axs[j][0], cmap=cm_strain, 
                    opacity=opacity_strain, clim=lim_strain)
            axs[j][0].set_title('Timestep '+str(i))

            # Plotting strain by x-value for each side
            max_strain = np.max(df.groupby(['X'])['Strain'].sum())

            for side, df in zip(['left', 'suture', 'right', 'asth'], [left_df, suture_df, right_df, asth_df]):
                
                # Quitting the loop if asthenosphere dataframe is empty
                if df.empty:
                    break

                # Sum strains along same x and reformat datatable
                strains_summed = df.groupby(['X']).sum() # note this accidentally sums the side&layer column too
                strains_summed_clipped = strains_summed[3e5:7e5+1]
                strains_summed_clipped.index /= 1000
                strains_summed_clipped = strains_summed_clipped.reindex(np.arange(300, 701), fill_value=0.0)

                # Subtract out the initial strain (if needed)
                # NOTE: may not work if initial timesteps lacks all 4 fields and later timesteps have them
                if subtract_initial:
                    # Getting initial strains (if needed)
                    if i == 0:
                        initial_df[side] += strains_summed_clipped['Strain']

                    strains_summed_clipped['Strain'] -= initial_df[side]

                # Isolate x values (km) and strains
                x_values = strains_summed_clipped.index
                y_values = strains_summed_clipped['Strain']
            
                '''
                # Use filter to smooth strains
                if len(y_values) > 25:
                    y_smoothed = savgol_filter(y_values,25,polyorder=3)
                else:
                    y_smoothed = y_values
            
                # Normalized smoothed strain using maximum strain value
                if max_strain != 0:
                    y_normalized = y_smoothed / max_strain
                else:
                    y_normalized = y_smoothed
                
                # Calculating the total strain in the side
                side_sum = str(int(np.sum(y_values)))

                # Calculating the total strain in the main part of the side
                positive_indices = np.asarray(y_normalized > 0).nonzero()[0]
                total_norm_strain = np.sum(y_normalized[positive_indices])
                    
                # Getting middle 90% of strain
                if total_norm_strain > 0:
                    cutoff = 0.10 * 0.5 * total_norm_strain
                    
                    curr_sum = 0
                    j = -1
                    while curr_sum < cutoff:
                        j += 1
                        curr_sum += y_normalized[positive_indices[j]]
                    left_index = positive_indices[j]

                    curr_sum = 0
                    j = len(positive_indices)
                    while curr_sum < cutoff:
                        j -= 1
                        curr_sum += y_normalized[positive_indices[j]]
                    right_index = positive_indices[j]
                
                    # Calculating values for diffusivity file
                    x1 = x_values[left_index]
                    x2 = x_values[right_index]
                    central_strain = np.sum(y_values[left_index:right_index + 1])

                    #print(side, x1, x2, central_strain)
                else:
                    left_index = 0
                    right_index = len(x_values) - 1
                    '''

                # Plotting
                y_label = None
                if side == 'left':
                    color = 'blue'
                    y_label = 'Left Zone'
                elif side == 'right':
                    color = 'green'
                    y_label = 'Right Zone'
                elif side == 'suture':
                    y_label = 'Central Zone'
                    color = 'orange'
                elif side == 'asth':
                    y_label = 'Asthenosphere'
                    color = 'purple'
                #y_label = side + ': ' + side_sum

                if (k == 0 or k == 4) and side != 'asth' and j != 0:
                    axs[j][1].plot(x_values,y_values, label=y_label, color=color)
                
                '''
                axs[i][2].plot(x_values,y_normalized)
                axs[i][3].plot(x_values[left_index:right_index + 1],y_normalized[left_index:right_index + 1])

                # Getting main part of strain based on max of strain
                max_side_strain = np.max(y_normalized)
                min_strain = 0.10 * max_side_strain
                valid_indices = (y_normalized > min_strain).astype(bool)
                alt_central_strain = y_values[valid_indices].sum()
                cumulative_width = len(y_normalized[valid_indices])
                

                axs[i][4].plot(x_values[valid_indices], y_normalized[valid_indices])
                '''

            # Formatting Time!
            if j == 2:
                axs[j][1].legend()

            axs[j][0].set_ylabel('y (km)')
            axs[j][0].set_ylim(500, axs[j][0].get_ylim()[1])
            axs[j][0].set_xlabel('x (km)')

            axs[j][1].set_ylabel('Noninitial Strain')
            axs[j][1].set_xlabel('x (km)')

            j += 1

    # More formatting
    y_min = 0.0
    y_max = 0.0
    for j in range(0, 3):
        y_min = min(y_min, axs[j][1].get_ylim()[0])
        y_max = max(y_max, axs[j][1].get_ylim()[1])

    for j in range(1, 3):
        axs[j][1].set_ylim(y_min, y_max)
        axs[j][1].axhline(0, color='black')

    axs[0, 1].axis("off")
    #handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
        
    fig.savefig(output_dir + str(k+1)+'_poster.png')
