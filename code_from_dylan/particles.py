"""
Module to analyze and post-process particles from ASPECT
"""
import os
import numpy as np
from scipy.spatial import KDTree
from joblib import Parallel,delayed
from numba import jit
from tqdm import tqdm

def nearest_neighbor_KDTree(position,positions):
    """
    Find nearest neighbor of a particle using X,Y,Z positions.

    Uses KDTree function from SciPy
    """
    # Get nearest neighbor from KDTree
    distance,index = KDTree(positions).query(position)

    return(index)

@jit
def nearest_neighbor_numpy(position,positions):
    """
    Find nearest neighbor of a particle using X,Y,Z positions.
    Uses pure Numpy to allow optimization with Numba
    """

    # Calculate distances

    distances = np.sqrt(
        (position[0]-positions[:,0])**2 + 
        (position[1]-positions[:,1])**2 +
        (position[2]-positions[:,2])**2
        )

    # Get index of the minimum distance
    index = np.argmin(distances)

    return(index)

def run_scalar_forward(source_mesh,future_meshes,field,interpolate=True,
                        method='KDTree',
                        processes=os.cpu_count()-6):
    """
    Apply scalar values on particles to same particles in future meshes
    """
    # Get ids of old particles
    old_particles = source_mesh['id']

    # Get scalars corresponding to those ids
    old_scalars = source_mesh[field]

    # Get positions for those ids
    old_positions = source_mesh.points

    print('Running Scalars Forward...')
    print('Processes: ',processes)

    for mesh in tqdm(future_meshes):
        
        # Get new particle ids
        new_particles = mesh['id']
        new_positions = mesh.points

        # Set up KDTree
        if method=='KDTree':
            tree = KDTree(old_positions)
            
            # Loop through new particles
            new_scalars = Parallel(n_jobs=processes,require='sharedmem')(
                delayed(get_previous_scalar_KDTree)(particle,new_positions[k],old_particles,old_scalars,
                    tree, interpolate=interpolate) 
                    for k,particle in enumerate(new_particles)
                ) 

        elif method=='numpy':
            # Loop through new particles
            new_scalars = Parallel(n_jobs=processes,require='sharedmem')(
                delayed(get_previous_scalar_numpy)(particle,new_positions[k],old_particles,old_scalars,
                    old_positions, interpolate=interpolate) 
                    for k,particle in enumerate(new_particles)
                ) 

        mesh[field] = new_scalars

        old_particles = new_particles
        old_scalars = np.array(new_scalars)
        old_positions = new_positions

    return(future_meshes)

def get_previous_scalar_numpy(particle,position,old_particles,old_scalars,
    old_positions,interpolate=False):
    """
    Get scalar value from previous timestep
    """
    # Try to get scalar from particles
    scalar = old_scalars[particle==old_particles]
    
    # Check if scalar actually exists
    if (scalar.size == 0) & (interpolate == False):
        new_scalar = np.nan
    
    elif (scalar.size == 0) & (interpolate == True):
        
        nn_index = nearest_neighbor_numpy(position,old_positions)
        
        # Assign scalar using nearest neighbor index
        new_scalar = float(old_scalars[nn_index])

    elif (scalar.size == 1):
        new_scalar = float(scalar)

    else:
        # Skip if duplicates of particle id
        new_scalar = np.nan

    return(new_scalar)        

def get_previous_scalar_KDTree(particle,position,old_particles,old_scalars,
    tree,interpolate=False):
    """
    Get scalar value from previous timestep
    """
    # Try to get scalar from particles
    scalar = old_scalars[particle==old_particles]
    
    # Check if scalar actually exists
    if (scalar.size == 0) & (interpolate == False):
        new_scalar = np.nan
    
    elif (scalar.size == 0) & (interpolate == True):
        
        # Find index of nearest neighbor
        distance,index = tree.query(position)
        
        # Assign scalar using nearest neighbor index
        new_scalar = float(old_scalars[index])

        # Try again if nan values returned
        attempt = 2
        while (np.isnan(new_scalar)) & (attempt<1000):
            distance,index = tree.query(position,k=[attempt])
            new_scalar = float(old_scalars[index])
            attempt = attempt+1

    elif (scalar.size == 1):
        new_scalar = float(scalar)

    else:
        # Skip if duplicates of particle id
        new_scalar = np.nan

    return(new_scalar)    

def subtract_scalar(mesh,subtract_mesh,field,new_field,interpolate=True,method='KDTree',
                        processes=os.cpu_count()-6):
    """
    Subtract scalar value in one mesh from another
    """
    # Get ids of old particles
    subtract_particles = subtract_mesh['id']

    # Get scalars corresponding to those ids
    subtract_scalars = subtract_mesh[field]

    # Get positions for those ids
    subtract_positions = subtract_mesh.points

    # Get ids of old particles
    mesh_particles = mesh['id']

    # Get scalars corresponding to those ids
    mesh_scalars = mesh[field]

    # Get positions for those ids
    mesh_positions = mesh.points

    # Set up KDTree
    if method=='KDTree':
        
        # Tree not used in following step
        tree = KDTree(mesh_positions)
        
        # Loop through new particles, without interpolation
        subtract_scalars_new = Parallel(n_jobs=processes,require='sharedmem')(
            delayed(get_previous_scalar_KDTree)(particle,mesh_positions[k],subtract_particles,subtract_scalars,
                tree, interpolate=False) 
                for k,particle in enumerate(mesh_particles)
            ) 

    elif method=='numpy':
        # Loop through new particles
        subtract_scalars_new = Parallel(n_jobs=processes,require='sharedmem')(
            delayed(get_previous_scalar_numpy)(particle,mesh_positions[k],subtract_particles,subtract_scalars,
                subtract_positions, interpolate=False) 
                for k,particle in enumerate(mesh_particles)
            ) 

    if interpolate==True:
        nan_positions = mesh_positions[np.array([subtract_scalars_new])==np.nan]

        distances,indices = tree.query(nan_positions)

    mesh[new_field] = mesh[field] - subtract_scalars_new

    return(mesh)


