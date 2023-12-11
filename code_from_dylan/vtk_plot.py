"""
Functions for plotting data from VTU/PVTU files.
"""
import os
import gc
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from tqdm import tqdm
from joblib import Parallel,delayed,dump,load
from scipy.spatial import KDTree
from matplotlib import cm,colors

from tchron import tchron as tc

def plot2D(file,field,bounds,ax=None,contours=False,colorbar=False,
         cfields=['crust_upper','crust_lower','mantle_lithosphere'],
         null_field='asthenosphere',contour_color='black',
         contours_only=False,**kwargs):
    """
    Plot 2D ASPECT results using Pyvista.

    Parameters
    ----------
    file : VTU or PVTU file to plot
    field : Field to use for color.
    bounds : List of bounds (km) by which to clip the plot.
    contours : Boolean for whether to add temperature contours. 
        The default is False.
    cfields : Names of compositional fields to use if field is 'comp_field.' 
        The default is ['crust_upper','crust_lower','mantle_lithosphere'].
    null_field : Null field if field is 'comp_field.'
        The default is 'asthenosphere'.

    Returns
    -------

    """
    
    mesh = pv.read(file)
    
    km2m = 1000
    bounds_m = [bound*km2m for bound in bounds] # Convert bounds to m
    bounds_3D = bounds_m + [0,0]
    mesh = mesh.clip_box(bounds=bounds_3D,invert=False)
    
    if field=='comp_field':
        mesh = comp_field_vtk(mesh,fields=cfields,null_field=null_field)
    
    if contours==True:
        cntrs = add_contours(mesh)
    
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True)
    sargs = dict(width=0.6,fmt='%.1e',height=0.2,label_font_size=32,
                 position_x=0.1)
    
    
    if contours_only==False:
        plotter.add_mesh(mesh,scalars=field,scalar_bar_args=sargs,**kwargs)
    
    if contours ==True:
        plotter.add_mesh(cntrs,color=contour_color,line_width=5)
    
    plotter.view_xy()
    
    if (colorbar==False)&(contours_only==False):
        plotter.remove_scalar_bar()
    
    plotter.enable_depth_peeling(10)


    # Calculate Camera Position from Bounds
    bounds_array = np.array(bounds_m)
    xmag = float(abs(bounds_array[1] - bounds_array[0]))
    ymag = float(abs(bounds_array[3] - bounds_array[2]))
    aspect_ratio = ymag/xmag
  
    plotter.window_size = (1024,int(1024*aspect_ratio))
    
    xmid = xmag/2 + bounds_array[0] # X midpoint
    ymid = ymag/2 + bounds_array[2] # Y midpoint
    zoom = xmag*aspect_ratio*1.875 # Zoom level - not sure why 1.875 works
    
    position = (xmid,ymid,zoom)
    focal_point = (xmid,ymid,0)
    viewup = (0,1,0)
    
    camera = [position,focal_point,viewup]
    # print(camera)
    
    plotter.camera_position = camera
    plotter.camera_set = True
    
    # Create image
    img = plotter.screenshot(transparent_background=True,
                             return_img=True)
    
    # Plot using imshow
    if ax is None:
        ax = plt.gca()
    
    ax.imshow(img,aspect='equal',extent=bounds)
    
    plotter.clear()
    pv.close_all()
    
    return(ax)

def add_colorbar(fig,vmin=None,vmax=None,cmap='viridis',location=[0.1,0.08,0.8,0.02],
                 orientation='horizontal',log=False,**kwargs):
    cax = fig.add_axes(location)
    
    if log==False:
        norm = colors.Normalize(vmin=vmin,vmax=vmax)
        
    if log==True:
        norm = colors.LogNorm(vmin=vmin,vmax=vmax)
    mappable = cm.ScalarMappable(norm=norm,cmap=cmap)
    cbar = plt.colorbar(mappable,cax=cax,orientation=orientation,**kwargs)
    
    return(cax)

def add_contours(mesh,field='T',values=np.arange(500,1700,200)):
    """
    Add contours to mesh in Pyvista.

    Parameters
    ----------
    mesh : Pyvista mesh object.
    field : Scalar in Pyvista mesh object to use for contours. The default is
        T.
    values : Values for the contours. The default is np.arange(500,1700,200).

    Returns
    -------
    cntrs: Pyvista mesh containing the contours.

    """
    contour_mesh = mesh.copy()
    cntrs = contour_mesh.contour(isosurfaces=values,scalars=field)
    return(cntrs)
    

def comp_field_vtk(mesh,fields=['crust_upper','crust_lower','mantle_lithosphere'],
               null_field='asthenosphere'):
    """
    Calculate compositional field from Pyvista VTK mesh.
    
    Uses input fields to assign value based on when fields are >0.9. Any point
    lacking a field >0.9 is assigned to the null field.

    Parameters
    ----------
    mesh : Pyvista mesh
    fields : Names of compositional fields that are scalars in the mesh.
        The default is ['crust_upper','crust_lower','mantle_lithosphere'].
    null_field : Name of field for points not included in compositional fields.
        The default is 'asthenosphere'.

    Returns
    -------
    mesh: Pyvista mesh with 'comp_field' added as a scalar.
    
    """

    # Create empty np array
    output = np.zeros(shape=mesh.point_data[fields[0]].shape)
    for x in range(len(fields)):
        array = mesh.point_data[fields[x]]
        output = np.where(array>0.5,x+1,output)
    
    mesh.point_data['comp_field'] = output
    return(mesh)

def He_age_vtk_parallel(files,system,time_interval,file_prefix='meshes_He',
               path='./',temp='~/dump',
               U=100,Th=100,radius=50,batch_size=100,processes=os.cpu_count()-2,
               He_profile_nodes=513,interpolate_profile=True,all_timesteps=True):
    """
    Function to do parallel He forward modeling of ASPECT VTK data.
    """
    dtype=np.float32
    
    print('Calculating He Ages...')
    if batch_size == 'auto':
        pre_dispatch = 2*processes
    else:
        pre_dispatch = 2*batch_size
    
    print('Processes: ',processes)
    print('Batch Size: ',batch_size)
    print('Pre-Dispatch: ',pre_dispatch)
    
    # Path for temporary memory dumps
    temp = os.path.expanduser(temp)
    
    try:
        shutil.rmtree(temp)
    except:
        pass
    
    os.makedirs(temp)
    
    new_dir = os.path.join(path,file_prefix)
    os.makedirs(new_dir,exist_ok=True)
    
    # Path for dump of cached profile
    cache_path = os.path.join(new_dir,'cache_profile.npy')
    
    with Parallel(n_jobs=processes,
                  batch_size=batch_size,
                  pre_dispatch=pre_dispatch,
                  temp_folder=temp) as parallel:
    
        # Loop through timesteps
        for k,file in enumerate(files):  
            
            filename = file_prefix+'_'+str(k)+'.vtu'
            filepath = os.path.join(new_dir,filename)
            
            # Check if timestep already exists
            if os.path.exists(filepath):
                print('Timestep ' + str(k) + ' Previously Run')
                
                next_filename = file_prefix+'_'+str(k+1)+'.vtu'
                next_filepath = os.path.join(new_dir,next_filename)
                
                # Check if this is the last file to exist
                if ~os.path.exists(next_filepath):
                    old_mesh = pv.read(filepath)
                    ids = old_mesh['id']
                    positions = old_mesh.points
                    new_profiles = np.load(cache_path)
                continue
                   
            mesh = pv.read(file)
            
            temps = mesh['T']
            
            if k==0:
                # Set up empty arrays for first timestep
                prof_shape = (len(temps),He_profile_nodes)
                old_profiles = np.empty(prof_shape,dtype=dtype)
                old_profiles.fill(np.nan)
                old_ids = np.ones(len(temps),dtype=dtype)*np.nan
                old_positions = np.ones(len(temps),dtype=dtype)*np.nan
            elif k>0:  
                old_profiles=new_profiles
                old_ids = ids # Get ids for previous profiles
                old_positions = positions
            
            gc.collect()
            if k in np.arange(5,len(files),5):
                shutil.rmtree(temp)
                os.makedirs(temp)
            
            ids = mesh['id']
            positions = mesh.points
        
            # Set up KDTree for timestep if doing interpolation
            if (k>0)&(interpolate_profile==True):
                
                # Get particle ids of particles with profiles
                hasprofile = ~np.isnan(old_profiles).all(axis=1)
                other_particles = old_ids[hasprofile]
                
                # Get positions of other particles
                other_positions = old_positions[hasprofile]
                
                # Set up KDTree to find closest particle
                tree = KDTree(other_positions)
                
            else:
                tree=None
                other_particles=None
                
            inputs = (k,positions,tree,ids,old_ids,temps,old_profiles,
                      U,Th,radius,time_interval,other_particles,
                      system,He_profile_nodes)
                
            # Calculate ages on last timestep only if indicated
            if all_timesteps==True:
                calc_age=True
            elif k==len(files)-1:
                calc_age=True
            else:
                calc_age=False
            
            print('Caluclating Profiles for Timestep ',k,'...')
            
    
            output = parallel(
                (delayed(particle_He_profile)
                 (particle,inputs,calc_age,interpolate_profile) 
                 for particle in tqdm(ids,position=0))
                )
            
            ages,new_profiles = zip(*output)
        
            # Convert new_profiles to array and save for reload
            new_profiles = np.array(new_profiles,dtype=dtype)
            np.save(cache_path,new_profiles)
        
            # Assign ages to mesh
            mesh.point_data[system] = np.array(ages,dtype=dtype)
        
            # Save new mesh
            mesh.save(filepath)
    
    # Delete cached profile when all finished
    os.remove(cache_path)
    
    return

def particle_He_profile(particle,inputs,calc_age,interpolate_profile):
    
    """
    Function to calculate He profile for a particular ASPECT particle.
    """
    # Use float32 to reduce memory usage
    dtype=np.float32
    
    # Unpack inputs
    (k,positions,tree,ids,old_ids,temps,old_profiles,
     U,Th,radius,time_interval,other_particles,
     system,He_profile_nodes) = inputs
    
    # Get old profile for current particle if present
    array = old_profiles[particle==old_ids]
     
    # If array is empty, assign np.nan
    if array.size == 0:
        profile = np.empty(He_profile_nodes,dtype=dtype)
        profile.fill(np.nan)

    # Otherwise, assign new value from old profile
    else:
        profile = array 
    
    # Get particle temperature
    particle_temp = temps[ids==particle]
       
    # If particle not found, don't attempt to calculate profile or age
    if particle_temp.size == 0:            
        x = np.empty(He_profile_nodes,dtype=dtype)
        x.fill(np.nan)
        age = np.nan
        output = (age,x)
        return(output)
    
    # Use previous He from nearest neighbor in previous timestep if none present
    
    if (k>0) & (np.all(np.isnan(profile))):
        
        if interpolate_profile==True:
        
            # Get particle position
            particle_position = positions[ids==particle]
            
            # Find closest particle
            distance,index = tree.query(particle_position)
            
            # Get id of closest particle
            neighbor_id = other_particles[index]
            
            # Get profile of closest particle
            profile = old_profiles[neighbor_id==old_ids]
        
        # If turned off, return original profile of np.nan
        elif interpolate_profile==False:
            x = np.empty(He_profile_nodes,dtype=dtype)
            x.fill(np.nan)
            age = np.nan
            output = (age,x)
            return(output)
    
    if calc_age==True:
        age,vol,pos,x = tc.forward_model(U,Th,radius,particle_temp,time_interval,system,
                             initial_He=profile.flatten(),calc_age=True,print_age=False,
                             nodes=He_profile_nodes)
        
        output = (age,x)
        return(output)
        
    else:    
        x = tc.forward_model(U,Th,radius,particle_temp,time_interval,system,
                             initial_He=profile.flatten(),calc_age=False,print_age=False,
                             nodes=He_profile_nodes)
        age = np.nan
        output = (age,x)
        return(output)
    
    

def particle_trace(meshes,timesteps,point,y_field,x_field='time',
                   plot_path=False,disable_tqdm=True):
    """
    Get single particle path over multiple timesteps from meshes generated
    by load_particle_meshes.
    
    By default, gets values for a single field over time. Can specify a 
    second field (x_field) to plot two parameters over time.
    
    Parameters
    ----------
    meshes: MultiBlock object from laod_particle_meshes function
    timesteps: NumPy array of timesteps to pull
    point: ID of particle to trace
    y_field: Particle property for y-axis
    x_field: Particle property for x-axis.
    bounds: List of bounds to clip model [xmin,xmax,ymin,ymax,zmin,zmax].
        Note that 0 indicates bottom
    plot_path: Whether to plot the x-field and y-field
    
    Returns
    -------
    point_df: Pandas dataframe with timesteps, y-field, and x-field if
        applicable.
    """
    
    x_point = []
    y_point = []
    
    # Get first and last values of timesteps
    first = timesteps[0]
    last = timesteps[-1]
    
    # Loop over files
    if disable_tqdm==False:
        print('Tracing Particles...')
      
    for k,mesh in enumerate(tqdm(meshes[first:last+1],disable=disable_tqdm)):
        
        ids = pv.point_array(mesh,'id') # Get particle ids
        y_vals = pv.point_array(mesh,y_field) # Get y field values
        
        if y_field=='position': # If y array is 3D position
            x_vals = y_vals[:,0] # Get x coordinates
            y_vals = y_vals[:,1] # Get y coordinates

            # Rename fields if plotting 2D position
            if (x_field == 'position') & (y_field == 'position'):
                x_field = 'x'
                y_field = 'y' 
                df = pd.DataFrame(
                {x_field:x_vals,y_field:y_vals},index=ids.astype(int))
        
        # Get x field values if not plotting timesteps or 2D position
        elif x_field!='time':
            x_vals = pv.point_array(mesh,x_field)
            df = pd.DataFrame(
                {x_field:x_vals,y_field:y_vals},index=ids.astype(int))
        else:
            df = pd.DataFrame({y_field:y_vals},index=ids.astype(int))
        
        # Extract values for specific points and add to lists
        point_vals = df.loc[point,:]
        y = point_vals[y_field]
        if x_field!='time':
            x = point_vals[x_field]
            x_point.append(x)
        y_point.append(y)
        
        # Reset position fields if needed
        if (x_field == 'x') & (y_field == 'y'):
            x_field = 'position'
            y_field = 'position'

    
    # Convert lists to dataframe
            
    # Rename fields if plotting 2D position
    if (x_field == 'position') & (y_field == 'position'):
        x_field = 'x'
        y_field = 'y'
    
    if x_field!='time':
        point_df = pd.DataFrame({x_field:x_point,y_field:y_point},index=timesteps)
    else:
        point_df = pd.DataFrame({y_field:y_point},index=timesteps)
   
       
    if plot_path is True:  
        if x_field!='time':
            point_df.plot(x_field,y_field)
            plt.xlabel(x_field)
            
            # Label timesteps
            for n in point_df.index:
                plt.text(x=point_df.loc[n,x_field],y=point_df.loc[n,y_field],
                     s=str(n))
        else:
            plt.plot(point_df.index,point_df[y_field])
            plt.xlabel('Timestep')
        plt.ylabel(y_field)
        plt.annotate('ID: '+str(point),xy=(0.1,0.9),xycoords='axes fraction')
        
        
    return(point_df)

def get_tt_path(all_ids,all_temps,point,disable_tqdm=True):
    
    # Loop over files
    if disable_tqdm==False:
        print('Finding Tt path...')
      
    tt = np.zeros(len(all_temps))
    
    for k,temps in enumerate(tqdm(all_temps,disable=disable_tqdm)):
        
        ids = all_ids[k] # Get particle ids
        
        # Get temp for particular id
        temp = temps[ids==point]
        
        tt[k] = temp
        
    return(tt)

def extract_temps_positions(meshes):
    
    all_ids = []
    all_temps = []
    all_positions = []
    for mesh in meshes:
        mesh_ids = mesh.point_data['id']
        mesh_temps = mesh.point_data['T']
        mesh_positions = mesh.points
        
        all_ids.append(mesh_ids)
        all_temps.append(mesh_temps)
        all_positions.append(mesh_positions)
    
    return(all_ids,all_temps,all_positions)
    

def get_pvtu(directory,timesteps,kind='solution'):
    """
    Get list of .pvtu files from directory and timesteps.
    
    Assumes files are named according to ASPECT output conventions and in a
    single directory. Can be used for standard solution or particle naming
    schemes.
    
    Parameters
    ----------
    directory: Path to directory contaning ASPECT pvtu files.
    timesteps: Integer or NumPy array of timesteps to pull
    kind: Whether to pull standard solution or particles.
    
    Returns
    -------
    files: List of file paths
    """    
    # Set up directory building blocks
    main = directory
    if kind=='solution':
        prefix = 'solution-00'
    if kind == 'particles':
        prefix = 'particles-00'
    suffix = '.pvtu'
    
    # Get file paths for all timesteps
    if type(timesteps)==int:
        timesteps_str = str(timesteps).zfill(3)
        files = os.path.join(main,prefix+timesteps_str+suffix)
    else:
        timesteps_str = [str(x).zfill(3) for x in timesteps.tolist()]
        files = [os.path.join(main,prefix+x+suffix) for x in timesteps_str]
    

    return(files)

def get_topography(directory,timesteps):
    main = directory
    prefix = 'topography.0'
    suffix = '00'
    
    if type(timesteps)==int:
        timesteps_str = str(int(timesteps/5)).zfill(2)
        files = os.path.join(main,prefix+timesteps_str+suffix)
    else:
        timesteps_str = [str(x/5).zfill(2) for x in timesteps.tolist()]
        files = [os.path.join(main,prefix + x + suffix) for x in timesteps_str]
        
    return(files)

def particle_positions(meshes,timestep,bounds=None):
    """
    Get ids and positions of all particles in a particular timestep.
    
    Parameters
    ----------
    meshes: MultiBlock object from load_particle_meshes
    timestep: Timestep from which to pull positions.
    
    Returns
    -------
    ids: NumPy array of particle ids
    positions: NumPy array of particle positions (X,Y,Z)
    """

    mesh = meshes[timestep]
    
    if bounds is not None:
        mesh = mesh.clip_box(bounds=bounds,invert=False)
    
    ids = pv.point_array(mesh,'id')
    positions = pv.point_array(mesh,'position')
    df = pd.DataFrame(data=positions,index=ids)
    return(df)

def load_particle_meshes(directory,timesteps,save=True,filename='meshes.vtm',
                         bounds=None,parallel=True,
                         processes = os.cpu_count()-6,kind='particles'):
    """
    Load particle meshes, clip, and save to avoid duplicate computation. This is
    computationally intensive for large meshes and may be preferred to do on 
    Stampede2.
    
    Parameters
    ----------
    directory: Path to directory contaning ASPECT pvtu files.
    timesteps: Integer or NumPy array of timesteps to pull.
    filename: Name of file to save clipped meshes to.
    bounds: Bounds by which to clip the model box (km)
    
    Returns
    -------
    meshes: MultiBlock object of clipped meshes for each timestep.
    """
    
    # Set up directory building blocks
    files=get_pvtu(directory,timesteps,kind=kind)
    

    if parallel == True:
        print('Loading and Clipping Meshes...')
        print('Processes: ',processes)
        
        mesh_list = Parallel(n_jobs=processes,require='sharedmem')(
            delayed(loadclip_parallel)(file,bounds) for file in tqdm(files)
            )
    
        meshes = pv.MultiBlock(mesh_list)
    
    else:
        meshes = pv.MultiBlock()
        for file in tqdm(files):
            mesh = pv.read(file) # Major computation to load this.
        
            # Clip mesh to save space
            if bounds is not None:
                km2m = 1000
                bounds_m = [bound*km2m for bound in bounds]
                mesh = mesh.clip_box(bounds=bounds_m,invert=False)
            
            meshes.append(mesh)
            
    if save == True:
        # Save clipped meshes as smaller file to work with
        meshes.save(filename)
        
    return(meshes)

def loadclip_parallel(file,bounds):
    mesh = pv.read(file)
    
    if bounds is not None:
        km2m = 1000
        bounds_m = [bound*km2m for bound in bounds]
        mesh = mesh.clip_box(bounds=bounds_m,invert=False)
    
    return(mesh)

def allmeshes_particles(meshes):
    """
    Get particle ids for particles that occur in all of a set of meshes
    
    Parameters
    ----------
    meshes: MultiBlock object of meshes
    
    Returns
    -------
    all_particles: NumPy array of all particles that occur in all input meshes.
    """
    all_particles = pv.point_array(meshes[0],'id')
    
    print('Finding particles that appear in all meshes...')
    for mesh in tqdm(meshes[1:-1]):
        mesh_particles = pv.point_array(mesh,'id')
        common_particles = np.intersect1d(all_particles,mesh_particles)
        all_particles = common_particles
        
    return(all_particles)

def get_surface_particles(mesh,topography,buffer=100):
    """
    Get particle ids for particles that are at the surface for a given timestep
    
    Parameters
    ----------
    mesh: Pyvista mesh
    topography: Directory pointing to topography file
    buffer: Number of meters below surface to include
    
    Returns
    -------
    all_particles: NumPy array of all particles that occur in all input meshes.
    """
    points = mesh.points
    ids = mesh.point_data['id']
    
    topo = pd.read_csv(topography,delimiter=' ',header=None,skiprows=1)
    
    surface_ids = []
    positions = []
    print('Finding Near Surface Particles...')
    for k,point in enumerate(tqdm(points)):
        part_id = ids[k]
        topo_point = np.interp(point[0],topo[0],topo[1])
        if point[1]>=(topo_point-buffer):
            surface_ids.append(part_id)
            positions.append(point)
    df = pd.DataFrame(positions,index=surface_ids,columns=['X','Y','Z'])
    
    return(df)

def surface_all_timesteps(meshes,end_mesh,topography,buffer=100):
    """
    Get particle ids for particles that are at the surface for a given mesh
    and are present in all meshes.
    
    Parameters
    ----------
    meshes: MultiBlock object of meshes
    end_mesh: Mesh to use for surface
    topography: Directory pointing to topography file
    buffer: Number of meters below surface to include
    
    Returns
    -------
    surface_all_time: NumPy array of all particles at surfae and in all meshes.
    """
    particles = allmeshes_particles(meshes)
    surface_particles = get_surface_particles(end_mesh,topography,buffer).index
    surface_all_time = np.intersect1d(particles,surface_particles)
    
    return(surface_all_time)
        
def pull_profile(file,field,x_pos='midpoint'):
    """
    Pull profile of scalar field from pvtu file.
    """
    mesh = pv.read(file)
    scalar = mesh.point_data[field]
    points = mesh.points
    
    x = points[:,0]
    y = points[:,1]
    
    if x_pos=='midpoint':
        # Use x midpoint for profile
        xmax = x.max()
        xmin = x.min()
        x_pos = (xmax-xmin)/2
    
    y_profile = y[x==x_pos]
    s_profile = scalar[x==x_pos]
    
    # Sort by y_value
    sort_indices = np.argsort(y_profile)
    y_sort = y_profile[sort_indices]   
    s_sort = s_profile[sort_indices]
    
    
    return(y_sort,s_sort)

