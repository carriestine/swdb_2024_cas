"""
Helper routines for extracting ccf information from skeletons and ccf
properties such as the "name" or acronym" for a given ccf id.

"""
from copy import deepcopy
from numpy import int64

import math
import numpy as np
import pandas as pd

CCF_PROPERTIES = [
    "acronym",
    "rgb_triplet",
    "id",
    'mesh',
    'mesh_filename',
    "name",
    "structure_id_path",
]


def get_ccf_property(atlas, ccf_id, ccf_property, depth=None, print_id=False):
    """
    Gets the value of the property for a given ccf id. For example, suppose
    that ccf_id=1000 and ccf_property="name", then this routine returns the
    name of the brain region corresponding to the ccf id.

    Parameters
    ----------
    atlas: brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas
        CCF atlas.
    ccf_id : int
        Numerical ID of a CCF region.
    ccf_property : str
        Property of a brain region that is stored in the ccf. The global
        variable "CCF_PROPERTIES" lists all of the properties.

    Returns
    -------
    str/float/int
        Value of the property for a given ccf id.

    """
    # Check that inputs are valid
    err_msg = ccf_property + " is not ccf property!"
    assert ccf_property in CCF_PROPERTIES, err_msg

    # Return property value
    if depth is not None:
        ccf_id = get_ccf_id_by_depth(atlas, ccf_id, depth)

    if ccf_id in atlas.structures:
        return atlas.structures[ccf_id][ccf_property]
    else:
        return "NaN"        


def get_ccf_id_by_depth(atlas, ccf_id, depth):
    """
    Gets the CCF ID corresponding to a specific depth in the structure
    hierarchy.

    Parameters
    ----------
    atlas: brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas
        CCF atlas.
    ccf_id : int
        Numerical ID of a CCF region.
    depth : int
        Depth level to access in the structure hierarchy. If 'depth' exceeds
        the available levels, the function returns the structure ID at the
        maximum available depth.

    Returns
    -------
    int
        CCF ID corresponding to a specific depth in the structure hierarchy.

    """
    if ccf_id in atlas.structures:
        structures = atlas.structures[ccf_id]["structure_id_path"]
        return structures[min(depth, len(structures) - 1)]
    else:
        return ccf_id


def get_ccf_ids(
    atlas, skel, compartment_type=None, vertex_type=None, depth=None
):
    """
    Gets the CCF IDs for specified vertices in a skeleton.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas
        CCF atlas.
    skel : meshparty.skeleton.Skeleton
        Skeleton to extract CCF IDs from.
    compartment_type : int/None, optional
        Compartment type of vertices to extract if "compartment_type" is 1, 2,
        or 3. If compartment_type is not one of these values, vertices from
        any compartment will be returned, provided they meet the condition
        specified by 'vertex_type'. The default is None.
    vertex_type : str/None
        Specifies the type of vertices to extract; this value must be either
        'branch_points' or 'end_points' if it is not None. If it is None, any
        vertex will be returned, provided it satisfies the condition specified
        by 'compartment_type'. The default is None.

    Returns
    -------
    list
        CCF IDs for specified vertices in a skeleton.

    """
    # Get ccf ids from "atlas"
    skel = deepcopy(skel)
    vertices = get_vertices(skel, compartment_type, vertex_type)
    ccf_ids = skel.vertex_properties["ccf"][vertices]

    # Return ccf ids
    if depth is not None:
        return [get_ccf_id_by_depth(atlas, ccf_id, depth) for ccf_id in ccf_ids]
    else:
        return ccf_ids


def get_vertices(skel, compartment_type, vertex_type):
    """
    Gets the subset of vertices specified by 'vertex_type' from 'skel' that
    belong to the specified compartment.

    Parameters
    ----------
    skel : meshparty.skeleton.Skeleton
        Skeleton to extract vertices from.
    compartment_type : int/None
        Compartment type of vertices to extract if "compartment_type" is 1, 2,
        or 3. If compartment_type is not one of these values, vertices from
        any compartment will be returned, provided they meet the condition
        specified by 'vertex_type'.
    vertex_type : str/None
        Specifies the type of vertices to extract; this value must be either
        'branch_points' or 'end_points' if it is not None. If it is None, any
        vertex will be returned, provided it satisfies the condition specified
        by 'compartment_type'.

    Returns
    -------
    list
        Subset of vertices that satisfy conditions specified by
        "compartment_type" and "vertex_type".

    """
    # Special Cases
    if compartment_type == 1:
        return [skel.root]
    elif not compartment_type and not vertex_type:
        return skel.vertex_properties['compartment'] != -1

    # General Cases
    if compartment_type and not vertex_type:
        return skel.vertex_properties['compartment'] == compartment_type
    else:
        assert vertex_type in ["branch_points", "end_points"]
        verts = skel.end_points if vertex_type == "end_points" else skel.branch_points
        if compartment_type:
            compartments = skel.vertex_properties['compartment']
            return [v for v in verts if compartments[v] == compartment_type]
        else:
            return verts


def get_connectivity_matrix(
    atlas, skels, compartment_type, binary=False, depth=None
):
    # Initializations
    ccf_ids_list = list()
    for skel in skels:
        ccf_ids_list.append(
            get_ccf_ids(
                atlas,
                skel,
                compartment_type=compartment_type,
                depth=depth,
                vertex_type="end_points",
            )
        )
    ccf_ids, cnts = np.unique(ccf_ids_list, return_counts=True)
    ccf_ids = ccf_ids[cnts > 5]
    cnts = cnts[cnts > 5]

    region_to_idx = dict({r: idx for idx, r in enumerate(regions[cnts > 10])})

    # Populate matrix
    matrix = np.zeros((len(skels), len(region_to_idx)))
    for i, ccf_ids in enumerate(ccf_ids_list):
        ccf_ids, cnts = np.unique(ccf_ids, return_counts=True)
        for j, ccf_id in enumerate(ccf_ids):
            if not math.isnan(ccf_id) and ccf_id in region_to_idx.keys():
                matrix[i, region_to_idx[ccf_id]] = cnts[j]
    idx_to_region = {idx: r for r, idx in region_to_idx.items()}
    return (matrix > 0 if binary else matrix), idx_to_region
