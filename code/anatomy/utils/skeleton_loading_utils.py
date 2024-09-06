from cloudfiles import CloudFiles
from cloudvolume import CloudVolume
from meshparty import skeleton

import boto3
import numpy as np
import os
import pandas as pd
import skeleton_plot as skelplot
import warnings

warnings.filterwarnings('ignore')

BUCKET = "aind-open-data"
LM_DATASET_KEYS = [
    "exaSPIM_609281_2022-11-03_13-49-18_reconstructions/precomputed",
    "exaSPIM_651324_2023-03-06_15-13-25_reconstructions/precomputed",
    "exaSPIM_653158_2023-06-01_20-41-38_reconstructions/precomputed",
    "exaSPIM_653980_2023-08-10_20-08-29_reconstructions/precomputed",
    "mouselight_reconstructions/precomputed",
]

# load a skeleton from cloud volume with at least radious and compartment
# vertex properties
def load_cv_skeleton(root_id: int, cv_obj: CloudVolume):
    """Loads skeleton from ID.
    
    Args:
        root_id: cell ID
        cv_obj: cloudvolume object
    
    Returns:
        sk: skeleton in meshparty.skeleton.Skeleton format
    """
    cv_sk = cv_obj.skeleton.get(root_id)
    sk = skeleton.Skeleton(
        cv_sk.vertices, 
        cv_sk.edges, 
        vertex_properties={'radius': cv_sk.radius, 'compartment': cv_sk.compartment},
        root=int(np.where(cv_sk.compartment==1)[0]), # Note: the root index is different between em and lm
        remove_zero_length_edges=False,
    )
    return sk

# -- Load specific skeletons for Supplemental Notebook --

def load_em_skeleton_as_meshwork(skeleton_id, data_root):
    # skeleton_id: the root id of one skeleton
    # data_root: path to the anatomy data for the current device
    cv_obj = CloudVolume(f"file://{data_root}/em_minnie65_v1078") 
    cv_sk = cv_obj.skeleton.get(skeleton_id) #load an example skeleton
    
    sk = skeleton.Skeleton(cv_sk.vertices, 
                       cv_sk.edges, 
                       vertex_properties={'radius': cv_sk.radius,
                                          'compartment': cv_sk.compartment}, 
                       root = len(cv_sk.edges), # the final edge is root
                       remove_zero_length_edges = False)

    conversion_factor = 1000
    
    return sk, conversion_factor

def load_lm_skeleton_as_meshwork(skeleton_id, data_root):
    # skeleton_id: the root id of one skeleton
    # data_root: path to the anatomy data for the current device
    cv_obj = CloudVolume(f"file://{data_root}/exaSPIM_609281_2022-11-03_13-49-18_reconstructions")
    cv_sk = cv_obj.skeleton.get(skeleton_id) #load an example skeleton
    
    sk = skeleton.Skeleton(cv_sk.vertices, 
                           cv_sk.edges, 
                           vertex_properties={'radius': cv_sk.radius,
                                              'compartment': cv_sk.compartment,
                                              'allenId': cv_sk.allenId}, 
                           root = 0, 
                           # root = len(sk_em.edges), # when the final edge is root
                           remove_zero_length_edges = False)

    conversion_factor = 1 #for LM (data in microns )
    
    return sk, conversion_factor


def load_em_skeleton_as_df(skeleton_id, data_root):
    # skeleton_id: the root id of one skeleton
    # data_root: path to the anatomy data for the current device
    input_directory = f"file://{data_root}/ccf_em_minnie65_v1078"
    cv_obj = CloudVolume(input_directory, use_https = True) # Initialize cloud volume
    cv_sk = cv_obj.skeleton.get(skeleton_id) #load an example skeleton
    
    sk = skeleton.Skeleton(cv_sk.vertices, 
                       cv_sk.edges, 
                       vertex_properties={'radius': cv_sk.radius,
                                          'compartment': cv_sk.compartment}, 
                       root = len(cv_sk.edges), # the final edge is root
                       remove_zero_length_edges = False)

    conversion_factor = 1000
    
    skel_df = pd.DataFrame({'vertex_xyz': [x for x in cv_sk.vertices],
                        'vertex_x': [x[0] for x in cv_sk.vertices],
                        'vertex_y': [x[1] for x in cv_sk.vertices],
                        'vertex_z': [x[2] for x in cv_sk.vertices],
                        'd_path_um': sk.distance_to_root / conversion_factor,
                        'compartment': cv_sk.compartment, 
                        'presyn_counts': cv_sk.presyn_counts, 
                        'presyn_size': cv_sk.presyn_size, 
                        'postsyn_counts': cv_sk.postsyn_counts, 
                        'postsyn_size': cv_sk.postsyn_size,})
    skel_df.index.names = ['vertex_index']
    
    return skel_df


def load_em_segmentprops_to_df(data_root):
    # data_root: path to the anatomy data for the current device
    input_directory = f"file://{data_root}/ccf_em_minnie65_v1078"
    cv_obj = CloudVolume(input_directory, use_https = True) # Initialize cloud volume
    
    cf = CloudFiles(cv_obj.cloudpath)
    
    # get segment info
    segment_properties = cf.get_json("segment_properties/info")

    segment_tag_values = np.array(segment_properties['inline']['properties'][1]['values'])

    segment_tags = np.array(segment_properties['inline']['properties'][1]['tags'])
    segment_tags_map = pd.Series(np.array(segment_properties['inline']['properties'][1]['tags']))
    segment_tags_map = segment_tags_map.to_dict()

    # map values to root id
    seg_df = pd.DataFrame({
        'nucleus_id': segment_properties['inline']['properties'][0]['values'],
        segment_properties['inline']['properties'][2]['id']: segment_properties['inline']['properties'][2]['values'],
        segment_properties['inline']['properties'][3]['id']: segment_properties['inline']['properties'][3]['values'],
        segment_properties['inline']['properties'][4]['id']: segment_properties['inline']['properties'][4]['values'], 
        segment_properties['inline']['properties'][5]['id']: segment_properties['inline']['properties'][5]['values'],
        'cell_type': segment_tag_values[:,0],
        'brain_area': segment_tag_values[:,1],
    },
        index=segment_properties['inline']['ids'])

    # map tags to root id
    seg_df['cell_type'] = seg_df.cell_type.replace(segment_tags_map)
    seg_df['brain_area'] = seg_df.brain_area.replace(segment_tags_map)

    return seg_df


# -- Load LM skeletons --
def load_lm_datasets(data_root):
    """
    Loads all of the light microscopy neurons across four exaspim datasets and
    and the mouse light dataset.

    Parameters
    ----------
    data_root : str
        path to the anatomy data for the current device

    Returns
    -------
    list[meshparty.skeleton.Skeleton]
        Skeletons that represent light microscopy neurons.

    """
    skels = list()
    print("Loading datasets...")
    for key in LM_DATASET_KEYS:        
        cv_dataset = CloudVolume(f"file://{data_root}/{key}")
        skels.extend(load_skeletons(cv_dataset, key, data_root))
        print("")
    return skels


def load_skeletons(cv_dataset, key, data_root):
    """
    Loads all skeletons from a cloudvolume dataset.

    Parameters
    ----------
    cv_dataset : CloudVolume
        Dataset that contains a set of skeletons.
    key : str
        Name of dataset containing skeletons to be loaded.
    data_root : str
        path to the anatomy data for the current device

    Returns
    -------
    list[meshparty.skeleton.Skeleton]
        Skeletons from cloudvolume dataset.

    """
    skels = list()
    skeleton_ids = get_skeleton_ids(key, data_root)
    for i, skel_id in enumerate(skeleton_ids):
        progress_bar(i + 1, len(skeleton_ids), process_id=key)
        skels.append(get_skeleton(cv_dataset, skel_id))
    return skels


def get_skeleton_ids(key, data_root):
    """
    Extracts skeleton ids from directory containing precomputed skeleton
    objects.

    Parameters
    ----------
    key : str
        Name of dataset containing skeletons to be loaded.
    data_root : str
        path to the anatomy data for the current device

    Returns
    -------
    list[int]
        Skeleton ids extracted from "skeleton_paths".

    """
    path = f"{data_root}/{key}/skeleton/"
    return [int(f) for f in os.listdir(path) if f.isnumeric()]


def get_skeleton(cv_dataset, skel_id):
    """
    Gets a skeleton from a cloudvolume dataset.

    Parameters
    ----------
    cv_dataset : CloudVolume
        Dataset to be read from.
    skel_id : int
        Single id of skeleton to be read.

    Returns
    -------
    meshparty.skeleton.Skeleton
        Skeleton from a cloudvolume dataset.

    """
    cv_skel = cv_dataset.skeleton.get(skel_id)
    skel = skeleton.Skeleton(
        cv_skel.vertices, 
        cv_skel.edges,
        remove_zero_length_edges=False,
        root=0,
        vertex_properties=set_vertex_properties(cv_skel),
    )
    return skel


def set_vertex_properties(cv_skel):
    """
    Sets the vertex properties of mesh party skeleton.

    Parameters
    ----------
    cv_skel : CloudVolume.skeleton
        Skeleton to be initialized as a meshparty skeleton.

    Returns
    -------
    dict
        Vertex properties of skeleton.

    """
    vertex_properties = {
        "ccf": cv_skel.allenId,
        "compartment": cv_skel.compartment,
        "radius": cv_skel.radius,
    }
    return vertex_properties


def number_of_samples():
    """
    Returns the number of samples in the light microscopy dataset.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Number of samples in the light microscopy dataset.

    """
    return len(LM_DATASET_KEYS)


def progress_bar(current, total, process_id=None, bar_length=50):
    """
    Reports the progress of completing some process.

    Parameters
    ----------
    current : int
        Current iteration of process.
    total : int
        Total number of iterations to be completed
    bar_length : int, optional
        Length of progress bar. The default is 50.

    Returns
    -------
    None

    """
    progress = int(current / total * bar_length)
    preamble = f"{process_id}:  " if process_id else ""
    bar = (
        f"{preamble}[{'=' * progress}{' ' * (bar_length - progress)}] {current}/{total}"
    )
    print(f"\r{bar}", end="", flush=True)
