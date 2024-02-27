"""
Script to figure out (and eventaully correct) issues with mesh values
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mpltern
from scipy.signal import savgol_filter,find_peaks,peak_widths
from scipy.stats import skew,mode
from scipy.spatial import KDTree
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
models = [models[3], models[7]]

#names = ['Model ' + str(x) for x in range(1,17)]
names = ["Model 3", "Model 7"]
nums = [3, 7]

output_dir = r'results/suture_points/'

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

# Create dataframe for final values
final = pd.DataFrame([],columns=['Localization','Symmetry','C-ness'])

# TODO: Figure out how to get the correct combo of directories
for k,model in enumerate(tqdm(models[0:])):

    '''
    # Get the appropriate pvtu file
    base_dir = r'/mnt/d459dc32-537b-41a9-9d32-483256cce117/riftinversion_production/'
    suffix = r'/output_ri_rift/solution'
    pvtu_dir = base_dir + model + suffix
    
    # Figure out appropriate timesteps
    tstep_invert = int((times[k]+invert_times[k])/tstep_interval)
    
    if model == '080122_rip_a':
        tstep_invert = 194
    if model =='080122_rip_e':
        tstep_invert = 395
    
    # Pull the file locations
    file = vp.get_pvtu(pvtu_dir,tstep_invert)
    mesh = pv.read(file)
    '''
    side_dir = r'figs/'
    file = side_dir + model + '/' + model + '_10.vtu'
    #file = side_dir + model + '/' + model + '_0.vtu'
    if model == '071322_rip':
        file = side_dir + model + '/' + model + '_9.vtu'
    mesh = pv.read(file)

    bounds_mesh = [bound*1e3 for bound in bounds] + [0,0]
    clipped = mesh.clip_box(bounds_mesh)

    # Finding if the nearest particles have different values for 'rift_side'
    particles = clipped.points
    print('Total particles:', len(particles))
    particle_tree = KDTree(particles)
    suture_indices = []

    for i in range(len(particles)):
        dists, indices = particle_tree.query(particles[i], k=2)
        nearest_id = indices[1]

        # Checking if the nearest particle is on the opposite side of the rift
        # This uses ceiling division by 3 to perform the check
        if -(clipped['rift_side'][nearest_id] // -3) != -(clipped['rift_side'][i] // -3):
            # Ensuring points are both in the lithosphere
            #if clipped['rift_side'][i] > 0 and clipped['rift_side'][nearest_id] > 0:
            if not np.isnan(clipped['rift_side'][nearest_id]) and not np.isnan(clipped['rift_side'][i]):
                if clipped['rift_side'][nearest_id] != 0 and clipped['rift_side'][i] != 0:
                    suture_indices.append(i)
            elif np.isnan(clipped['rift_side'][i]):
                print("NaN detected")

            
    print(len(suture_indices))
    clipped['rift_side'][suture_indices] = 7

    # Plotting results (to ensure accuracy)
    fig,ax = plt.subplots(1,figsize=(8.5,11),dpi=300)
    ax.scatter(particles[suture_indices, 0], particles[suture_indices, 1])
    plt.savefig(output_dir + str(nums[k] + 1) + "_suture_no_nan_or_asth" + ".pdf")



    '''


    # Get all strain values
    x = clipped.points[:,0]
    x_rounded = np.round(x,-3)
    
    # Pull strains, sides, and x positions into dataframe
    strains = clipped[field]
    sides = clipped['rift_side']
    df = pd.DataFrame([x_rounded,strains,sides]).T
    df.columns = ['X','Strain', 'Side&Layer']

    
    # Splitting the table into left and right sides of the suture
    left_df = df[df['Side&Layer'] <= 3]
    right_df = df[df['Side&Layer'] >= 4]

    # Setting up plot
    fig,axs = plt.subplots(3,figsize=(8.5,11),dpi=300)
    #pv.start_xvfb()
    
    # Plot suture results and strain
    vp.plot2D(file,'rift_side',bounds=bounds,ax=axs[0])
    vp.plot2D(file,field,bounds,ax=axs[0], cmap=cm_strain, 
              opacity=opacity_strain, clim=lim_strain)
    axs[0].set_title('Model '+str(k+1))

    # Plotting strain by x-value for each side
    max_strain = np.max(df.groupby(['X'])['Strain'].sum())

    for side, df in zip(['left', 'right'], [left_df, right_df]):
        print(k)
        print(df.head())

        # Sum strains along same x and clip
        strains_summed = df.groupby(['X']).sum()
        strains_summed_clipped = strains_summed[3e5:7e5+1]
        
        # Isolate x values (km) and strains
        x_values = strains_summed_clipped.index/1000
        y_values = strains_summed_clipped['Strain']
        
        # Use filter to smooth strains
        y_smoothed = savgol_filter(y_values,25,polyorder=3)
        
        # Normalized smoothed strain using maximum strain value
        y_normalized = y_smoothed / max_strain               #np.max(y_smoothed)
        print('\nModel ',k+1)
        print('Side: ', side)

        axs[1].plot(x_values,y_values)
        axs[2].plot(x_values,y_normalized)
        #axs[2].scatter(x_peaks,heights,c='red')
        #axs[2].hlines(y=width_heights,xmin=min_x_peaks,xmax=max_x_peaks,color='red')
    
        #axs[3].scatter(x_peaks,symmetry_corrected)
        #axs[3].set_xlim(300,700)
        #axs[3].set_ylim(0,1)
        #axs[3].scatter([400,600],[percentile25,percentile75])
         
    plt.tight_layout()
        
    fig.savefig(output_dir + str(k+1)+'_strain_sides_results.pdf')
    '''
