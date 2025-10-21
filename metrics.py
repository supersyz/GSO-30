# metrics.py

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh

def eval_pointcloud(pre_mesh_file, gt_mesh_file, samplepoint, eval_type, 
                   thresholds=np.linspace(1./1000, 1, 1000)):
    """
    Evaluate Similarity between predicted pointclouds and gt pointclouds.

    samplepoint: num of samplepoints
    eval_type: (real_obj', 'syn_obj', 'syn_scene', .etc)
    thresholds: F-score's shreshold array
    return: dict
    """
    if eval_type == 'real_obj':
        gt_mesh = trimesh.load(gt_mesh_file, force='mesh')
        pointcloud_tgt, normals_tgt = sampleGT(gt_mesh, samplepointsnum=samplepoint)
        
        pre_mesh = trimesh.load(pre_mesh_file, force='mesh')
        pointcloud, normals = sampleGT(pre_mesh, samplepointsnum=samplepoint)
    else:
        gt_mesh = trimesh.load(gt_mesh_file, force='scene')
        pointcloud_tgt, normals_tgt = sampleGT(gt_mesh, samplepointsnum=samplepoint)
        
        pre_mesh = trimesh.load(pre_mesh_file, force='scene')
        pointcloud, normals = sampleGT(pre_mesh, samplepointsnum=samplepoint)

    if pointcloud.shape[0] == 0 or pointcloud_tgt.shape[0] == 0:
        out_dict = {
            'N_Acc' : 0,
            'N_Comp' : 0,
            'normals': 0,
            'CD_Acc' : 0,
            'CD_Comp': 0,
            'chamfer-L2': 0,
            'F_Acc_005': 0,
            'F_Comp_005': 0,
            'f-score-005': 0, 
            'F_Acc_03': 0,
            'F_Comp_03': 0,
            'f-score-03': 0, 
            'F_Acc_5': 0,
            'F_Comp_5': 0,
            'f-score-5': 0,   
        }
        return out_dict
    
    # Calculate Completeness
    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud, normals
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Calculate Accuracy
    accuracy, accuracy_normals = distance_p2p(
        pointcloud, normals, pointcloud_tgt, normals_tgt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Calculate Chamfer Distance
    normals_correctness = (
        0.5 * completeness_normals + 0.5 * accuracy_normals
    )
    chamferL2 = 0.5 * (completeness + accuracy)

    # Calculate F-Score
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i]) 
        if (precision[i] + recall[i]) > 0 else 0 
        for i in range(len(precision))
    ]

    out_dict = {
        'N_Acc' : accuracy_normals,
        'N_Comp' : completeness_normals,
        'normals': normals_correctness,
        'CD_Acc' : accuracy,
        'CD_Comp': completeness,
        'chamfer-L2': chamferL2,
        'F_Acc_005': precision[4],
        'F_Comp_005': recall[4],
        'f-score-005': F[4],       
        'F_Acc_03': precision[29],
        'F_Comp_03': recall[29],
        'f-score-03': F[29],       
        'F_Acc_1': precision[99],
        'F_Comp_1': recall[99],
        'f-score-1': F[99], 
        'F_Acc_2': precision[199], 
        'F_Comp_2': recall[199],
        'f-score-2': F[199],        
        'F_Acc_5': precision[499],
        'F_Comp_5': recall[499],
        'f-score-5': F[499],        
    }
    return out_dict

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' 
    Calculate dist and normal similarity between gt and tgt pointclouds's respectively
    Args:
        points_src (numpy array): (N, 3)
        normals_src (numpy array): (N, 3)
        points_tgt (numpy array): (M, 3)
        normals_tgt (numpy array): (M, 3)
    
    Returns:
        dist (numpy array): Nearest Distance, (N,)
        normals_dot_product (numpy array): Similarity of normals, (N,)
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src, p=2, k=1)   

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def get_threshold_percentage(dist, thresholds):
    ''' 
    Calculate the percentage of points which are lower than the threshold.
    Args:
        dist (numpy array): Caluated Distance, the shape is (N,)
        thresholds (numpy array): Threshold array (K,)
    
    Returns:
        in_threshold (list): percentage under specific thresholds, the length is K
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold

def sampleGT(mesh, samplepointsnum):
    ''' 
    Sample points and normals from mesh.
    
    Args:
        mesh (trimesh.Trimesh)
        samplepointsnum (int):
    
    Returns:
        sample_random (numpy array): (samplepointsnum, 3)
        sample_normal (numpy array): (samplepointsnum, 3)
    '''
    mesh.fix_normals()
    sample_random, index_random = trimesh.sample.sample_surface(mesh, samplepointsnum)
    sample_normal = mesh.face_normals[index_random]
    return sample_random, sample_normal
