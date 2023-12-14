#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 10:54:25 2022

@author: dyvasey
"""
import numpy as np
import matplotlib.pyplot as plt

from gdmate.io import pyvista_io
from gdmate.analysis_modules import particles
from gdmate.visualization import pyvista_vis


models = ['063022_rip_c','071822_rip_b','070422_rip_e','072022_rip_a',
          '070422_rip_c','071322_rip','070622_rip_a','072022_rip_b']

names = ['Slow/Cold Half-Breakup','Slow/Cold Half-Breakup w/ Cooling',
         'Slow/Cold Full-Breakup','Slow/Cold Full-Breakup w/ Cooling',
         'Hot/Fast Half-Breakup','Hot/Fast  Half-Breakup w/ Cooling',
         'Hot/Fast  Full-Breakup','Hot/Fast Full-Breakup w/ Cooling']

# Indicate time of final rift of reach model (post-cooling)

times = [16,36,32,52,7.3,27.3,14.5,34.5]
tstep_interval = 0.1

base_dir = r'/mnt/15c59731-2c7b-420d-8e97-048239b4d9c8/riftinversion_overflow/'
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
        kind='particles',processes=4)
    
    fields = ['initial crust_upper','initial crust_lower','initial mantle_lithosphere']

    base_mesh = meshes[0]
    conditions = [base_mesh[field]>0.5 for field in fields]
    
    asth = ~conditions[0] & ~conditions[1] & ~conditions[2]
    
    asth_points = base_mesh.points[asth]
    
    asth_y_values = asth_points[:,1]
    
    asth_max_indices = np.argsort(asth_y_values)[-100000:]
    
    asth_max_x_values = asth_points[:,0][asth_max_indices]
    asth_max_x_avg = np.median(asth_max_x_values)
    
    side = base_mesh.points[:,0] <= asth_max_x_avg
    
    left_side_upper = side & conditions[0]
    left_side_lower = side & conditions[1]
    left_side_mantle = side & conditions[2]
    
    right_side_upper = ~side & conditions[0]
    right_side_lower = ~side & conditions[1]
    right_side_mantle = ~side & conditions[2]
    
    values = np.zeros(left_side_upper.shape)
    values[left_side_upper] = 1
    values[left_side_lower] = 2
    values[left_side_mantle] = 3
    values[right_side_upper] = 4
    values[right_side_lower] = 5
    values[right_side_mantle] = 6
    
    base_mesh['rift_side'] = values
    
    new_meshes = particles.run_scalar_forward(meshes[0],meshes[1:],field='rift_side',
    processes=8,interpolate=True,method='KDTree')
    
    fig,axs = plt.subplots(int(len(meshes)),dpi=300,figsize=(8.5,2*len(meshes)))

    for n,ax in enumerate(axs):
        pyvista_vis.pv_plot_2d(meshes[n],'rift_side',bounds=bounds[0:4],ax=ax)
    
    fig.suptitle(names[k])
        
    meshes.save(r'figs/'+model+'.vtm')
    fig.savefig(r'figs/'+model+'.pdf')
