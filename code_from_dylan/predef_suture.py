#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 27 2024

@author: Peter Scully (based on code from Dylan Vasey)
"""
import numpy as np
import matplotlib.pyplot as plt

import pyvista_io
import particles
import pyvista_vis
import pyvista as pv


models = ['063022_rip_c','071822_rip_b', '070422_rip_e', '072022_rip_a',
          '070422_rip_c','071322_rip','070622_rip_a','072022_rip_b']
#models = ['072022_rip_b']
#models = ['063022_rip_c']

names = ['Slow/Cold Half-Breakup', 'Slow/Cold Half-Breakup w/ Cooling',
         'Slow/Cold Full-Breakup','Slow/Cold Full-Breakup w/ Cooling',
        'Hot/Fast Half-Breakup','Hot/Fast  Half-Breakup w/ Cooling',
         'Hot/Fast  Full-Breakup','Hot/Fast Full-Breakup w/ Cooling']
#names = ['Hot/Fast Full-Breakup w/ Cooling']
#names = ['Slow/Cold Half-Breakup']

# Indicate time of final rift of reach model (post-cooling)

times = [16,36,32,52,7.3,27.3,14.5,34.5]
#times = [34.5]
#times = [16]

tstep_interval = 0.1

base_dir = r'/mnt/d459dc32-537b-41a9-9d32-483256cce117/riftinversion_production/'
suffix = r'/output_ri_rift/particles'

bounds = [300e3,700e3,500e3,620e3,0,0]

for k,model in enumerate(models):
    
    # Get the appropriate pvtu file
    directory = base_dir + model + suffix
    
    start = int(times[k]/tstep_interval)
    
    end = int((times[k]+20)/tstep_interval)
    #end = start+3

    tsteps = np.arange(start,end+1,20)
    
    meshes = pyvista_io.pv_load_clipped_meshes(directory,tsteps,bounds=bounds,
        kind='particles',processes=2)
    
    fields = ['initial crust_upper','initial crust_lower','initial mantle_lithosphere']

    base_mesh = meshes[0]
    conditions = [base_mesh[field]>0.5 for field in fields]
    
    asth = ~conditions[0] & ~conditions[1] & ~conditions[2]
    
    asth_points = base_mesh.points[asth]
    
    asth_y_values = asth_points[:,1]
    
    asth_max_indices = np.argsort(asth_y_values)[-50000:]
    

    # Finding dimensions of initial "suture"
    min_lith_thickness = 600000.0 - np.max(asth_y_values)

    suture_width = (100.0 * 1000.0 * 1000.0) / min_lith_thickness

    print("thickness: ", min_lith_thickness)
    print("width: ", suture_width)

    # Finding cutoff points for suture
    asth_max_x_values = asth_points[:,0][asth_max_indices]
    
    asth_max_x_avg = np.median(asth_max_x_values)

    left_cutoff = asth_max_x_avg - suture_width / 2.0
    right_cutoff = asth_max_x_avg + suture_width / 2.0

    print("Left: ", left_cutoff)
    print("Right: ", right_cutoff)
    
    # Assigning sides 
    left_side = base_mesh.points[:,0] <= left_cutoff
    right_side = base_mesh.points[:,0] >= right_cutoff
    
    left_side_upper = left_side & conditions[0]
    left_side_lower = left_side & conditions[1]
    left_side_mantle = left_side & conditions[2]
    
    right_side_upper = right_side & conditions[0]
    right_side_lower = right_side & conditions[1]
    right_side_mantle = right_side & conditions[2]

    suture_zone_upper = ~left_side & ~right_side & conditions[0]
    suture_zone_lower = ~left_side & ~right_side & conditions[1]
    suture_zone_mantle = ~left_side & ~right_side & conditions[2]
    
    values = np.zeros(left_side_upper.shape)
    values[left_side_upper] = 1
    values[left_side_lower] = 2
    values[left_side_mantle] = 3
    values[right_side_upper] = 4
    values[right_side_lower] = 5
    values[right_side_mantle] = 6
    values[suture_zone_upper] = 7
    values[suture_zone_lower] = 8
    values[suture_zone_mantle] = 9
    
    base_mesh['rift_side'] = values
    
    new_meshes = particles.run_scalar_forward(meshes[0],meshes[1:],field='rift_side',
    processes=36,interpolate=True,method='KDTree')
    
    fig,axs = plt.subplots(int(len(meshes)),dpi=300,figsize=(8.5,2*len(meshes)))

    pv.start_xvfb()
    for n,ax in enumerate(axs):
        pyvista_vis.pv_plot_2d(meshes[n],'rift_side',bounds=bounds[0:4],ax=ax)
    
    fig.suptitle(names[k])
        
    meshes.save(r'predef_suture_figs/'+model+'.vtm')
    fig.savefig(r'predef_suture_figs/'+model+'.pdf')
