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

for k,model in enumerate(tqdm(models[0:])):

    # Setting up variables for plotting loop
    side_dir = r'predef_suture_figs/'

    last_mesh_num = 10
    if model == '071322_rip':
        last_mesh_num = 9
    bounds_mesh = [bound*1e3 for bound in bounds] +[0,0]

    # Setting up figure
    # Currently does NOT create normalized plot
    fig, axs = plt.subplots(last_mesh_num + 1, 2, dpi=300, figsize=(11, 17))

    # Setting up dataframe for noninitial strain plotting
    initial_df = pd.DataFrame()
    initial_df['left'] = np.zeros(701)
    initial_df['right'] = np.zeros(701)
    initial_df['suture'] = np.zeros(701)
    initial_df['asth'] = np.zeros(701)

    # Making plots for each timestep
    for i in range(0, last_mesh_num + 1):
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
        vp.plot2D(file,'rift_side',bounds=bounds,ax=axs[i][0])
        vp.plot2D(file,field,bounds,ax=axs[i][0], cmap=cm_strain, 
                opacity=opacity_strain, clim=lim_strain)
        axs[i][0].set_title('Model '+str(k+1))

        # Plotting strain by x-value for each side
        max_strain = np.max(df.groupby(['X'])['Strain'].sum())

        for side, df in zip(['left', 'right', 'suture', 'asth'], [left_df, right_df, suture_df, asth_df]):  
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
            
            y_label = side + ': ' + str(int(np.sum(y_values)))

            axs[i][1].plot(x_values,y_values, label=y_label)
            #axs[i][2].plot(x_values,y_normalized)
         
        axs[i][1].legend()

    plt.tight_layout()
        
    fig.savefig(output_dir + str(k+1)+'_noninitial_strain_sides_time.pdf')
