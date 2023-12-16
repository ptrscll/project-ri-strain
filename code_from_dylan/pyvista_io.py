"""
Module for Pyvista input and output
"""
import os
from tqdm import tqdm
from joblib import Parallel,delayed
import pyvista as pv

def pv_read(file,**kwargs):
    """
    Wrapper for pyvista.read
    """
    mesh = pv.read(file,**kwargs)
    return(mesh)

def pv_clip(mesh,bounds,invert=False,**kwargs):
    """
    Wrapper for pyvista.clip_box
    """
    clipped = mesh.clip_box(bounds=bounds,invert=invert,**kwargs)

    return(clipped)

def pv_clip_scalar(mesh,field,value,invert=False,**kwargs):
    """
    Wrapper for pyvista.clip_scalar
    """
    clipped = mesh.clip_scalar(field,value=value,invert=invert,**kwargs)

    return(clipped)

def pv_get_pvtu(directory,timesteps,kind='solution'):
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

def pv_read_clip(file,bounds=None,field=None,value=None):
    """Combine pv_read and pv_clip into single function for parallel compution"""
    mesh = pv_read(file)
    
    if field is not None:
        mesh = pv_clip_scalar(mesh,field=field,value=value)
    
    if bounds is not None:
        mesh = pv_clip(mesh,bounds=bounds)
    
    return(mesh)
    

def pv_load_clipped_meshes(directory,timesteps,bounds=None,field=None,
                           value=None,save=False,
                           filename='meshes.vtm',parallel=True,
                           processes = os.cpu_count()-6,kind='solution'):
    """
    Load meshes, clip, and save to avoid duplicate computation. 
    
    This is computationally intensive for large meshes and may be preferred to do on 
    Stampede2. By default, uses parallel computation from joblib.
    
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
    files=pv_get_pvtu(directory,timesteps,kind=kind)
    

    if parallel == True:
        print('Loading and Clipping Meshes...')
        print('Processes: ',processes)
        
        mesh_list = Parallel(n_jobs=processes,require='sharedmem')(
            delayed(pv_read_clip)(file,bounds,field,value) for file in tqdm(files)
            )
    
        meshes = pv.MultiBlock(mesh_list)
    
    else:
        meshes = pv.MultiBlock()
        for file in tqdm(files):
            mesh = pv_read_clip(file,bounds)            
            meshes.append(mesh)
            
    if save == True:
        # Save clipped meshes as smaller file to work with
        meshes.save(filename)
        
    return(meshes)
