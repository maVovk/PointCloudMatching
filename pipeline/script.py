import os
import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import open3d as o3d
from igraph import *
import open3d.t.pipelines.registration as treg
from tqdm.notebook import tqdm

from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation
import time
import copy

import gc

preprocess_time = []
graph_construction_time = []
max_clique_time = []
transform_time = []
mac_cpp_time = []

def visualize_iou(aligned_grid):
    plt.imshow(aligned_grid)
    plt.axis(False)
    plt.grid(True, alpha=0.5, linestyle='-')

    plt.show()

def make_pcd_from_array(array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)

    return pcd

def remove_floor_and_ceil(cloud, floor_height=-0.9, ceil_height=1.5):
    heights = np.linspace(-4.0, 4.0, 41)
    floor_index = None
    if floor_height == 'auto':
        bins = []
        for i, height in enumerate(heights[:-1]):
            bins.append(len(cloud[(cloud[:, 2] > height) * (cloud[:, 2] < heights[i + 1])]))
        #print('Bins:', bins)
        floor_index = np.argmax(bins[:20]) + 1
        floor_height = heights[floor_index]
        assert floor_index < len(heights) - 5
    if ceil_height == 'auto':
        if floor_index is None:
            floor_index = 0
            while floor_index < len(heights) - 6 and heights[floor_index] < floor_height:
                floor_index += 1
        ceil_index = floor_index + 5 + np.argmax(bins[floor_index + 5:])
        ceil_height = heights[ceil_index]
    #print('Floor height:', floor_height)
    #print('Ceil height:', ceil_height)
    return cloud[(cloud[:, 2] > floor_height) * (cloud[:, 2] < ceil_height)]

def raycast(grid, n_rays=1000, center_point=None):
    grid_raycasted = grid.copy()
    if center_point is None:
        center_point = (grid.shape[0] // 2, grid.shape[1] // 2)
    resolution = 0.1
    radius = 8.0 / resolution
    for sector in range(n_rays):
        angle = sector / n_rays * 2 * np.pi - np.pi
        ii = center_point[0] + np.sin(angle) * np.arange(0, radius)
        jj = center_point[1] + np.cos(angle) * np.arange(0, radius)
        ii = ii.astype(int)
        jj = jj.astype(int)
        good_ids = ((ii > 0) * (ii < grid.shape[0]) ** (jj > 0) * (jj < grid.shape[1])).astype(bool)
        ii = ii[good_ids]
        jj = jj[good_ids]
        points_on_ray = grid[ii, jj]
        if len(points_on_ray.nonzero()[0]) > 0:
            last_obst = points_on_ray.nonzero()[0][-1]
            grid_raycasted[ii[:last_obst], jj[:last_obst]] = 1
        #else:
            #grid_raycasted[ii, jj] = 1
    return grid_raycasted

def transform_grid(grid, x, y, theta):
    shift_vector = np.array([x, y])
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    center = np.array(grid.shape) / 2
    offset = center - rotation_matrix @ center
    offset += shift_vector

    grid_transformed = np.zeros(grid.size).reshape(grid.shape)
    grid_transformed = affine_transform(grid, rotation_matrix, offset=offset, order=0, mode='constant', cval=0)
    grid_transformed = grid_transformed.astype(np.uint8)
    return grid_transformed

def transform_pcd(cloud, x, y, theta):
    pcd_transformed = cloud.copy()  # эта функция нигде не вызывается, оставляем так
    return pcd_transformed

def get_occupancy_grid(points_xyz, resolution=0.1, radius=18, clip=8):
    index = np.isnan(points_xyz).any(axis=1)
    
    points_xyz = np.delete(points_xyz, index, axis=0)
    points_xyz = points_xyz[(points_xyz[:, 0] > -clip) * (points_xyz[:, 0] < clip) * \
                            (points_xyz[:, 1] > -clip) * (points_xyz[:, 1] < clip)]
    points_xyz_obstacles = remove_floor_and_ceil(points_xyz, floor_height=-0.3, ceil_height=0.5)

    #print('Points xyz:', points_xyz.shape, points_xyz[0], points_xyz.min(), points_xyz.max())
    grid = np.zeros((int(2 * radius / resolution), int(2 * radius / resolution)), dtype=np.uint8)
    points_ij = np.round(points_xyz[:, :2] / resolution).astype(int) + \
                        [int(radius / resolution), int(radius / resolution)]
    points_ij = points_ij[(points_ij[:, 0] >= 0) * (points_ij[:, 0] < grid.shape[0]) * \
                          (points_ij[:, 1] >= 0) * (points_ij[:, 1] < grid.shape[1])]
    
    grid[points_ij[:, 0], points_ij[:, 1]] = 1
    grid = raycast(grid)
    
    points_ij = np.round(points_xyz_obstacles[:, :2] / resolution).astype(int) + \
                [int(radius / resolution), int(radius / resolution)]
    points_ij = points_ij[(points_ij[:, 0] >= 0) * (points_ij[:, 0] < grid.shape[0]) * \
                          (points_ij[:, 1] >= 0) * (points_ij[:, 1] < grid.shape[1])]
    
    grid[points_ij[:, 0], points_ij[:, 1]] = 2
    return grid

def get_iou_of_grids(rel_x, rel_y, rel_theta, cur_grid, v_grid, visualize, save=False, cnt=0):
    rel_x_rotated = -rel_x * np.cos(rel_theta) - rel_y * np.sin(rel_theta)
    rel_y_rotated = rel_x * np.sin(rel_theta) - rel_y * np.cos(rel_theta)
    rel_x, rel_y = rel_x_rotated, rel_y_rotated
    
    cur_grid_transformed = transform_grid(cur_grid, rel_x, rel_y, rel_theta)
    cur_grid_transformed[cur_grid_transformed > 0] = 1
    
    v_grid_copy = v_grid.copy()
    v_grid_copy[v_grid_copy > 0] = 1
    
    intersection = np.sum(v_grid_copy * cur_grid_transformed)
    union = np.sum(v_grid_copy | cur_grid_transformed)

    grid_aligned = np.zeros((v_grid.shape[0], v_grid.shape[1], 3))
    grid_aligned[:, :, 0] = cur_grid_transformed
    grid_aligned[:, :, 1] = v_grid
    grid_aligned = (grid_aligned * 255).astype(np.uint8)
    
    if visualize:
        visualize_iou(grid_aligned)

    return intersection / union

def get_iou(rel_x, rel_y, rel_theta, cur_cloud, v_cloud, visualize=False):
    cur_grid = get_occupancy_grid(cur_cloud)
    v_grid = get_occupancy_grid(v_cloud)

    try:
        return get_iou_of_grids(rel_x, rel_y, rel_theta, cur_grid, v_grid, visualize)
    except ZeroDivisionError:
        return 0
    except Exception as e:
        print(e)
        return 0

def normalize(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def transformation_error(gt_pose_shift, tf_matrix_estimated):
    tf_rotation = Rotation.from_matrix(tf_matrix_estimated[:3, :3]).as_rotvec()
    gt_x, gt_y, gt_angle = gt_pose_shift
    #gt_angle = -gt_angle
    estimated_angle = tf_rotation[2]
    estimated_x = tf_matrix_estimated[0, 3]
    estimated_y = tf_matrix_estimated[1, 3]
    return (np.abs(estimated_x - gt_x), np.abs(estimated_y - gt_y), np.abs(normalize(estimated_angle - gt_angle)))

def get_rel_pose(x1, y1, theta1, x2, y2, theta2):
    return [x1 - x2, y1 - y2, theta1 - theta2]

def euclidean(a, b):
    return torch.norm(a - b, dim=-1, keepdim=True)

def compatibility(a,b):
    assert(a.shape[-1]==6)
    assert(b.shape[-1]==6)
    n1=torch.norm(a[...,:3]-b[...,:3],dim=-1,keepdim=True)
    n2=torch.norm(a[...,3:]-b[...,3:],dim=-1,keepdim=True)
    return torch.abs(n1-n2)

def Dmatrix(a,type):
    if type=="euclidean":
        return torch.cdist(a,a)
        
    elif type=="compatibility":
        a1=a[...,:3]
        a2=a[...,3:]
        return torch.abs(Dmatrix(a1,"euclidean")-Dmatrix(a2,"euclidean"))

class GraphConstructor(nn.Module):
    def __init__(self,inlier_thresh,thresh,trainable,device="cuda",sigma=None,tau=None) -> None:
        '''
        inlier thresh: KITTI 0.6, 3dmatch 0.1
        thresh: fpfh 0.9, fcgf 0.999
        '''
        super().__init__()
        self.device=device
        self.inlier_thresh=nn.Parameter(torch.tensor(inlier_thresh,requires_grad=trainable,dtype=torch.float32)).to(device)
        self.thresh=nn.Parameter(torch.tensor(thresh,requires_grad=trainable,dtype=torch.float32)).to(device)
        if sigma is not None:
            self.sigma=nn.Parameter(torch.tensor(sigma,requires_grad=trainable,dtype=torch.float32)).to(device)
        else:
            self.sigma=self.inlier_thresh
        if tau is not None:
            self.tau=nn.Parameter(torch.tensor(tau,requires_grad=trainable,dtype=torch.float32)).to(device)
        else:
            self.tau=self.thresh
    def forward(self,points,mode,k1=1,k2=1):
        '''
        points: B x M x 6
        output: B x M x M
        '''
        if mode=="correspondence":
            points=points.to(self.device)
            dmatrix=Dmatrix(points,"compatibility")
            score=1-dmatrix**2/self.inlier_thresh**2
            # score=torch.exp(-dmatrix**2/self.inlier_thresh**2)
            score[score<self.thresh]=0
            if k1==1:
                return score
            else:
                return score*torch.einsum("bmn,bnk->bmk",score,score)
        elif mode=="pointcloud":
            '''
            points: B x N x 3
            output: B x N x N
            '''
            points=points.to(self.device)
            dmatrix=Dmatrix(points,"euclidean")
            
            # score=1-dmatrix**2/self.inlier_thresh**2
            score=torch.exp(-dmatrix**2/self.sigma**2)
            score[score<self.tau]=0
            if k2==1:
                return score
            else:
                return score*torch.einsum("bmn,bnk->bmk",score,score)
            
def extract_fpfh_features(pcd, downsample):
    keypts = pcd.voxel_down_sample(downsample)
    keypts.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=downsample * 2, max_nn=30))
    
    features = o3d.pipelines.registration.compute_fpfh_feature(keypts, o3d.geometry.KDTreeSearchParamHybrid(
        radius=downsample * 5, max_nn=100))
    features = np.array(features.data).T
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)

    return keypts, features

def preprocess_MAC(source, target, voxel_size):
    t1 = time.perf_counter()
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source)

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target)

    src_kpts, src_desc = extract_fpfh_features(src_pcd, voxel_size)
    tgt_kpts, tgt_desc = extract_fpfh_features(tgt_pcd, voxel_size)

    distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
    min_vals = np.min(distance, axis=1)
    source_idx = np.argmin(distance, axis=1)
    corr_idx = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]],
                            axis=-1)

    source_points = torch.from_numpy(np.array(src_kpts.points)).type(torch.FloatTensor)[corr_idx[:,0]]
    target_points = torch.from_numpy(np.array(tgt_kpts.points)).type(torch.FloatTensor)[corr_idx[:,1]]
    # source_points = np.array(src_kpts.points)[corr_idx[:,0]]
    # target_points = np.array(tgt_kpts.points)[corr_idx[:,1]]
    correspondence_pcd = np.hstack([source_points, target_points])

    preprocess_time.append(time.perf_counter() - t1)
    return source_points, target_points, correspondence_pcd, corr_idx, min_vals


def construct_graph(correspondences):
    pts = torch.from_numpy(correspondences).type(torch.FloatTensor)
    pts=pts.to('cuda')

    t1 = time.perf_counter()
    gc = GraphConstructor(0.6, 0.9, False, 'cuda', 0.6, 0.)

    corr_graph = gc(pts, 'correspondence')
    FCG = corr_graph
    FCG = FCG - torch.diag_embed(torch.diag(FCG))
    FCG[FCG < 0.999] = 0
    SCG = torch.matmul(FCG, FCG) * FCG

    graph_construction_time.append(time.perf_counter() - t1)
    return SCG.cpu().detach().numpy()

def find_cliques(TOP_G):
    t1 = time.perf_counter()
    graph = Graph.Adjacency((TOP_G > 0).tolist())
    graph.es['weight'] = TOP_G[TOP_G.nonzero()]
    graph.vs['label'] = range(0, TOP_G.shape[0])
    graph.to_undirected()

    print(f'Num of edges: {graph.ecount()}', end=' ', flush=True)

    if graph.ecount() > 120000:
        return None

    macs = graph.maximal_cliques(min=3)

    if graph.ecount() > 90000:
        print('found')

    max_clique_time.append(time.perf_counter() - t1)
    return macs

def transform(pts, trans):
    if len(pts.shape) == 3:
        trans_pts = torch.einsum('bnm,bmk->bnk', trans[:, :3, :3],
                                 pts.permute(0, 2, 1)) + trans[:, :3, 3:4]
        return trans_pts.permute(0, 2, 1)
    else:
        trans_pts = torch.einsum('nm,mk->nk', trans[:3, :3],
                                 pts.T) + trans[:3, 3:4]
        return trans_pts.T

def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans

def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)  # 升维度，然后变为对角阵
    H = Am.permute(0, 2, 1) @ Weight @ Bm  # permute : tensor中的每一块做转置

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)

def post_refinement(initial_trans, src_kpts, tgt_kpts, iters, inlier_threshold = 0.1, weights=None):
    pre_inlier_count = 0
    for i in range(iters):
        pred_tgt = transform(src_kpts, initial_trans)
        L2_dis = torch.norm(pred_tgt - tgt_kpts, dim=-1)
        pred_inlier = (L2_dis < inlier_threshold)[0]
        inlier_count = torch.sum(pred_inlier)
        if inlier_count <= pre_inlier_count:
            break
        pre_inlier_count = inlier_count
        initial_trans = rigid_transform_3d(
            A=src_kpts[:, pred_inlier, :],
            B=tgt_kpts[:, pred_inlier, :],
            weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier]
        )
    return initial_trans


def calculate_transform(source_points, target_points, correspondences, TOP_G, macs, inlier_threshold=0.1):
    t1 = time.perf_counter()

    clique_weight = np.zeros(len(macs), dtype=float)
    for ind in range(len(macs)):
        mac = list(macs[ind])
        if len(mac) >= 3:
            for i in range(len(mac)):
                for j in range(i + 1, len(mac)):
                    clique_weight[ind] = clique_weight[ind] + TOP_G[mac[i], mac[j]]

    clique_ind_of_node = np.ones(correspondences.shape[0], dtype=int) * -1
    max_clique_weight = np.zeros(correspondences.shape[0], dtype=float)
    max_size = 3
    for ind in range(len(macs)):
        mac = list(macs[ind])
        weight = clique_weight[ind]
        if weight > 0:
            for i in range(len(mac)):
                if weight > max_clique_weight[mac[i]]:
                    max_clique_weight[mac[i]] = weight
                    clique_ind_of_node[mac[i]] = ind
                    max_size = len(mac) > max_size and len(mac) or max_size

    filtered_clique_ind = list(set(clique_ind_of_node))
    try:
        filtered_clique_ind.remove(-1)
    except:
        pass

    group = []
    for _ in range(3, max_size + 1):
        group.append([])
    for ind in filtered_clique_ind:
        mac = list(macs[ind])
        group[len(mac) - 3].append(ind)

    tensor_list_A = []
    tensor_list_B = []
    for i in range(len(group)):
        if len(group[i]) == 0:
            continue
        batch_A = source_points[list(macs[group[i][0]])][None]
        batch_B = target_points[list(macs[group[i][0]])][None]
        if len(group) == 1:
            continue
        for j in range(1, len(group[i])):
            mac = list(macs[group[i][j]])
            src_corr = source_points[mac][None]
            tgt_corr = target_points[mac][None]
            batch_A = torch.cat((batch_A, src_corr), 0)
            batch_B = torch.cat((batch_B, tgt_corr), 0)
        tensor_list_A.append(batch_A)
        tensor_list_B.append(batch_B)

    max_score = 0
    final_trans = torch.eye(4)
    for i in range(len(tensor_list_A)):
        trans = rigid_transform_3d(tensor_list_A[i], tensor_list_B[i], None, 0)
        pred_tgt = transform(source_points[None], trans)  # [bs,  num_corr, 3]
        L2_dis = torch.norm(pred_tgt - target_points[None], dim=-1)  # [bs, num_corr]
        MAE_score = torch.div(torch.sub(inlier_threshold, L2_dis), inlier_threshold)
        MAE_score = torch.sum(MAE_score * (L2_dis < inlier_threshold), dim=-1)
        max_batch_score_ind = MAE_score.argmax(dim=-1)
        max_batch_score = MAE_score[max_batch_score_ind]
        if max_batch_score > max_score:
            max_score = max_batch_score
            final_trans = trans[max_batch_score_ind]


    final_trans1 = post_refinement(final_trans[None], source_points[None], target_points[None], 20, inlier_threshold)

    final_trans = final_trans1[0]
    final_trans = final_trans.cpu().numpy()

    transform_time.append(time.perf_counter() - t1)
    return final_trans


if __name__ == '__main__':
    dataset_dir = './data'
    data_dir = './fastmac_data'
    counter = 0

    pr_data_dir = os.path.join(dataset_dir, 'place_recognition_data')
    graph_data_dir = os.path.join(dataset_dir, 'graph_data')

    ious = []
    dists = []
    registration_results = []
    calculated_dirs = []
    infinite = ['1710504173985918000', '1710503900111192000', '1710506770508046000']

    with open(os.path.join(data_dir, 'tested_dirs1.txt')) as file:
        calculated_dirs = [line.strip() for line in file]

    registration_results = list(np.loadtxt(os.path.join(data_dir, 'result.txt')))
    all_times = np.loadtxt(os.path.join(data_dir, 'mean_times.txt'))

    # if len(all_times) != 0:
    #     preprocess_time += list(all_times)[::4]
    #     graph_construction_time += list()[1::4]
    #     max_clique_time += list(all_times)
    #     transform_time += list(all_times[:, 3])

    print(len(set(calculated_dirs)))

    for ii, test_name in tqdm(list(enumerate(os.listdir(pr_data_dir)))):
        if int(test_name) in calculated_dirs:
            continue

        if test_name in infinite:
            print('infinite')
            continue

        test_dir = os.path.join(pr_data_dir, test_name)
        try:
            transforms_ = np.loadtxt(os.path.join(test_dir, 'transforms.txt'))
        except FileNotFoundError:
            continue

        if transforms_.size == 0:
            continue
        if transforms_.ndim == 1:
            transforms_ = transforms_[np.newaxis, :]

        cand_clouds = {}
        cand_poses = {}

        print(test_name, end=' ', flush=True)

        # Reference cloud (from robot)
        with np.load(os.path.join(test_dir, 'cloud.npz')) as CloudFile:
            ref_cloud = CloudFile['arr_0']

        ref_cloud = ref_cloud[:, :3]
        ref_cloud = ref_cloud[ref_cloud == ref_cloud].reshape((-1, 3))

        if transforms_.size == 0:
            pass

        for idx in transforms_[:, 0]:
            cloud = np.load(os.path.join(graph_data_dir, str(int(idx)), 'cloud.npz'))['arr_0']
            pose_stamped = np.loadtxt(os.path.join(graph_data_dir, str(int(idx)), 'pose_stamped.txt'))
            cand_clouds[int(idx)] = cloud
            cand_poses[int(idx)] = pose_stamped[1:]

        gt_poses = np.loadtxt(os.path.join(test_dir, 'gt_poses.txt'))

        for i in range(transforms_.size // 7):
            idx = transforms_[i, 0]
            gt_pose = cand_poses[int(idx)]
            pose_shift = get_rel_pose(*gt_poses[0], *gt_pose)

            # Candidate cloud (from graph of locations)
            cloud = cand_clouds[idx][:, :3]
            cloud = cloud[cloud == cloud].reshape((-1, 3))

            pose_shift = get_rel_pose(*gt_poses[0], *gt_pose)
            dist = np.sqrt(pose_shift[0] ** 2 + pose_shift[1] ** 2)
            iou = get_iou(*pose_shift, ref_cloud, cloud, visualize=False)

            voxel_size = 0.2

            print(f'IoU: {iou:.2f}', end=' ', flush=True)

            source_points, target_points, correspondences, _, _ = preprocess_MAC(ref_cloud, cloud, voxel_size)

            TOP_G = construct_graph(correspondences)
            macs = find_cliques(TOP_G)

            if macs is None:
                _ = preprocess_time.pop()
                _ = graph_construction_time.pop()
                continue

            transformation_matrix = calculate_transform(source_points, target_points, correspondences, TOP_G, macs)
            # source_points, target_poio

            er = transformation_error(pose_shift, transformation_matrix)
            evaluation = o3d.pipelines.registration.evaluate_registration(make_pcd_from_array(ref_cloud), make_pcd_from_array(cloud), 0.9, transformation_matrix)
            registration_results.append([evaluation.fitness, *er, iou])

            all_times = np.append(all_times, [preprocess_time[-1], graph_construction_time[-1], max_clique_time[-1], transform_time[-1]])

        calculated_dirs.append(test_name)
        np.savetxt(os.path.join(data_dir, 'tested_dirs1.txt'), np.array(calculated_dirs), fmt='%s')
        np.savetxt(os.path.join(data_dir, 'result.txt'), np.array(registration_results))
        np.savetxt(
            os.path.join(data_dir, 'mean_times.txt'),
            all_times
        )
        # np.savetxt(
        #     os.path.join(data_dir, 'median_times.txt'),
        #     np.array([np.median(preprocess_time), np.median(graph_construction_time), np.median(max_clique_time), np.median(transform_time)])
        # )
        counter += 1
        print(counter, flush=True)

        if counter >= 30:
            gc.collect()
            counter = 0