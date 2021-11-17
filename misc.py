from __future__ import print_function

import torch
import numpy as np

from programs.utils import draw_vertical_leg as draw_vertical_leg_new
from programs.utils import draw_rectangle_top as draw_rectangle_top_new
from programs.utils import draw_square_top as draw_square_top_new
from programs.utils import draw_circle_top as draw_circle_top_new
from programs.utils import draw_middle_rect_layer as draw_middle_rect_layer_new
from programs.utils import draw_circle_support as draw_circle_support_new
from programs.utils import draw_square_support as draw_square_support_new
from programs.utils import draw_circle_base as draw_circle_base_new
from programs.utils import draw_square_base as draw_square_base_new
from programs.utils import draw_cross_base as draw_cross_base_new
from programs.utils import draw_sideboard as draw_sideboard_new
from programs.utils import draw_horizontal_bar as draw_horizontal_bar_new
from programs.utils import draw_vertboard as draw_vertboard_new
from programs.utils import draw_locker as draw_locker_new
from programs.utils import draw_tilt_back as draw_tilt_back_new
from programs.utils import draw_chair_beam as draw_chair_beam_new
from programs.utils import draw_line as draw_line_new
from programs.utils import draw_back_support as draw_back_support_new

from programs.loop_gen import decode_loop, translate, rotate, end

from scipy.ndimage import zoom
import pytorch3d
from pytorch3d.ops import cubify
import open3d as o3d
import os
import time

def scale_voxels(voxelgrid, scale):
    """LeoZDong addition: Scale (down) voxels to make the longest side shorter.
    shape2prog trained with a cuztom voxelization where the longest side 24, but 
    ShapeNet's default voxelization has longest side 32.
    Args:
        voxelgrid (np.ndarray): (32, 32, 32)
        scale (float): less than 1 downscale factor
    """
    assert scale < 1
    scaled_voxel = (zoom(voxelgrid, scale) > 0.5).astype(int)

    # Pad with zeros to match the old size
    old_size = voxelgrid.shape[0]
    new_size = scaled_voxel.shape[0]
    pad_size = old_size - new_size
    assert pad_size % 2 == 0
    pad_size //= 2
    scaled_voxel = np.pad(scaled_voxel, pad_size)

    return scaled_voxel


def get_segpoints_labels_batch(segpoints, voxelgrid_labels, mesh_center, mesh_size,
                               vox_side_len=24):
    segpoints_labels = []

    for i in range(len(segpoints)):
        t = time.time()
        segpoints_labels_i = get_segpoints_labels(segpoints[i],
                                                  voxelgrid_labels[i],
                                                  mesh_center[i],
                                                  mesh_size[i],
                                                  vox_side_len)
        segpoints_labels.append(segpoints_labels_i)
        # print("segpoints labels one shape time:", round(time.time() - t, 3))

    return np.stack(segpoints_labels, 0)


def get_segpoints_labels(segpoints, voxelgrid_labels, mesh_center, mesh_size,
                         vox_side_len=24):
    """LeoZDong addition: Query segmentation point labels based on labeled 
    voxelgrid. This is for a single item.
    Args:
        segpoints (ndarray): (npoints, 3)
        voxelgrid_labels (ndarray): (32, 32, 32) -1 for empty voxel and other 
            numbers for part / primitive ids.
        mesh_center (float)
        mesh_size (float)
    Returns: (npoints, ) Labeled ID for each segpoint.
    """
    # Un-rotate voxels to pointcloud orientation
    voxelgrid_labels = voxelgrid_labels.copy()
    voxelgrid_labels = np.flip(voxelgrid_labels, 1)
    voxelgrid_labels = np.swapaxes(voxelgrid_labels, 0, 1)
    # voxelgrid = np.swapaxes(voxelgrid, 0, 2)

    # Extract voxelgrid from voxelgrid_labels
    voxelgrid = (voxelgrid_labels >= 0).astype(int)

    # Find voxel centers as if they are in a [-0.5, 0.5] bbox
    vox_center = get_vox_centers(voxelgrid)

    # Rescale points so that the mesh object is 0-centered and has longest side
    # to be 0.75 (vox_side_len=24 / 32)
    segpoints += vox_center - mesh_center
    scale = (vox_side_len / voxelgrid.shape[0]) / mesh_size
    segpoints *= scale

    # Initialize labels to -1 (no label)
    segpoints_labels = np.ones((len(segpoints),)) * -1
    in_bounds_cond = np.stack((segpoints.min(1) > -0.5, segpoints.max(1) < 0.5), 0)
    in_bounds = np.all(in_bounds_cond, 0)
    # Can use the same function to find voxel label instead of voxel occ
    in_bounds_labels = points_occ_in_voxel(voxelgrid_labels, segpoints[in_bounds, :])
    segpoints_labels[in_bounds] = in_bounds_labels

    # Deal with points that do not fall into occupied blocks
    assigned_idx = np.where(segpoints_labels >= 0)
    unassigned_idx = np.where(segpoints_labels < 0)
    if len(unassigned_idx[0]) == 0:
        return segpoints_labels
    if len(assigned_idx[0]) == 0:
        print("Misses completely!")
        return np.zeros((len(segpoints), ))
    # For each unassigned point, find its nearest neighbor among assigned points
    # and set its label to that assigned point's label.
    assigned_points = segpoints[assigned_idx]
    unassigned_points = segpoints[unassigned_idx]
    pairwise_dists = np.linalg.norm(unassigned_points[:, None, :] - assigned_points[None, :, :],
                                    axis=-1)
    try:
        nearest_assigned_point_idx = assigned_idx[0][np.argmax(-pairwise_dists, 1)]
    except:
        import ipdb; ipdb.set_trace()
    segpoints_labels[unassigned_idx] = segpoints_labels[nearest_assigned_point_idx]

    return segpoints_labels


def voxel_to_aligned_mesh(voxelgrid, target_sizes, target_centers):
    """LeoZDong addition: Convert voxels to mesh with rescaling and recentering.
    Args:
        target_sizes (torch.Tensor): (N,) Size of the longest side of the mesh after rescaling.
        target_centers (torch.Tensor): (N, 3) Center of the mesh after aligning.
    """
    # Un-rotate the voxels to shapenet voxel rotation
    voxelgrid = voxelgrid.copy()
    voxelgrid = np.flip(voxelgrid, 2)
    voxelgrid = np.swapaxes(voxelgrid, 1, 2)

    # Important note: occnet pointcloud is not aligned with shapenet voxel!
    # I need to do `gt_pc = torch.stack((z, y, x), 1)` to get the occnet
    # pointcloud in same orientation as the voxel
    # Need to do additional step to align with occnet pointcloud
    voxelgrid = np.swapaxes(voxelgrid, 1, 3)

    # Deal with empty voxelgrid predictions
    for i, vox in enumerate(voxelgrid):
        if vox.sum() == 0:
            voxelgrid[i, 0, 0] = 1

    # import ipdb; ipdb.set_trace()
    voxelgrid = torch.tensor(voxelgrid.copy())
    meshes = cubify(voxelgrid, 0.5, align="center")
    bbox = meshes.get_bounding_boxes()

    vox_mesh_centers = (bbox[:, :, 1] + bbox[:, :, 0]) / 2
    offsets = target_centers - vox_mesh_centers
    offsets_packed = []
    for i, verts in enumerate(meshes.verts_list()):
        offsets_packed.append(offsets[i].repeat(len(verts), 1))

    offsets_packed = torch.cat(offsets_packed, 0)

    mesh_sizes = (bbox[:, :, 1] - bbox[:, :, 0]).max(1).values
    scales = target_sizes / mesh_sizes

    meshes.offset_verts_(offsets_packed)
    meshes.scale_verts_(scales)

    return meshes


def vox_mesh_iou(voxelgrid, mesh_size, mesh_center, points, points_occ, vox_side_len=24, pc=None):
    """LeoZDong addition: Compare iou between voxel and mesh (represented as
    points sampled uniformly inside the mesh). Everything is a single element 
    (i.e. no batch dimension).
    """
    # Un-rotate voxels to pointcloud orientation
    voxelgrid = voxelgrid.copy()
    voxelgrid = np.flip(voxelgrid, 1)
    voxelgrid = np.swapaxes(voxelgrid, 0, 1)
    # voxelgrid = np.swapaxes(voxelgrid, 0, 2)

    # Find voxel centers as if they are in a [-0.5, 0.5] bbox
    vox_center = get_vox_centers(voxelgrid)

    # Rescale points so that the mesh object is 0-centered and has longest side
    # to be 0.75 (vox_side_len=24 / 32)
    points += vox_center - mesh_center
    scale = (vox_side_len / voxelgrid.shape[0]) / mesh_size
    points *= scale

    # import ipdb; ipdb.set_trace()
    cond = np.stack((points.min(1) > -0.5, points.max(1) < 0.5), 0)
    in_bounds = np.all(cond, 0)
    vox_occ = np.zeros_like(points_occ)
    vox_occ[in_bounds] = points_occ_in_voxel(voxelgrid, points[in_bounds, :])

    # Find occupancy in voxel for the query points
    # vox_occ = points_occ_in_voxel(voxelgrid, points)
    iou = occ_iou(points_occ, vox_occ)

    #### DEBUG ####
    # vox_occ_points = points[vox_occ > 0.5]
    # gt_occ_points = points[points_occ > 0.5]
    # int_occ_points = points[(vox_occ * points_occ) > 0.5]

    # save_dir = '/viscam/u/leozdong/shape2prog/output/chair/GA_24/meshes/table/cd5f235344ff4c10d5b24cafb84903c7'
    # save_ply(vox_occ_points, os.path.join(save_dir, 'vox_occ_points.ply'))
    # save_ply(gt_occ_points, os.path.join(save_dir, 'gt_occ_points.ply'))
    # save_ply(int_occ_points, os.path.join(save_dir, 'int_occ_points.ply'))

    # print("iou:", iou)
    return iou

def save_ply(pointcloud, save_file):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.io.write_point_cloud(save_file, pcd)

def vox_mesh_iou_batch(voxelgrids, mesh_sizes, mesh_centers, points, points_occ, vox_side_len=24, pc=None):
    ious = []
    for i in range(len(voxelgrids)):
        voxelgrid = voxelgrids[i]
        mesh_size = mesh_sizes[i]
        mesh_center = mesh_centers[i]
        points_i = points[i]
        points_occ_i = points_occ[i]
        iou = vox_mesh_iou(voxelgrid,
                           mesh_size,
                           mesh_center,
                           points_i,
                           points_occ_i,
                           vox_side_len)
        ious.append(iou)

    return ious


def points_occ_in_voxel(voxel, points):
    """LeoZDong addition: Get point occupancy for a single voxel grid.
    Args:
        voxel (np.ndarray): (grid_size, grid_size, grid_size)
        points (np.ndarray): (n_points, 3)
    """
    # Get voxel indices that each point falls in
    vox_grid_size = voxel.shape[0]
    vox_idx = np.floor((points + 0.5) * vox_grid_size).astype(int)

    # Clamp because sometimes numerical instability gets a vox_idx rounded to 32
    vox_idx = np.clip(vox_idx, 0, 31).astype(int)

    points_occ = voxel[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]]

    return points_occ


def occ_iou(occ1, occ2):
    """Computes the Intersection over Union (IoU) value for two sets of occupancy
    values.
    Adopted from Occupancy Networks repo.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    """
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IoU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def get_sizes_and_centers(pointcloud):
    """LeoZDong addition: Calculate target sizes and centers from target pointcloud."""
    x_max = pointcloud[:, :, 0].max(1).values
    x_min = pointcloud[:, :, 0].min(1).values
    x_sizes = x_max - x_min
    x_centers = (x_max + x_min) / 2

    y_max = pointcloud[:, :, 1].max(1).values
    y_min = pointcloud[:, :, 1].min(1).values
    y_sizes = y_max - y_min
    y_centers = (y_max + y_min) / 2

    z_max = pointcloud[:, :, 2].max(1).values
    z_min = pointcloud[:, :, 2].min(1).values
    z_sizes = z_max - z_min
    z_centers = (z_max + z_min) / 2

    sizes = torch.stack((x_sizes, y_sizes, z_sizes), 1).max(1).values
    centers = torch.stack((x_centers, y_centers, z_centers), 1)

    return sizes, centers

def get_vox_centers(voxelgrid):
    """Find the voxel center as if it is in a [-.5, .5] bbox.
    This applies to a single voxelgrid, i.e. no batch dimension.
    """
    if voxelgrid.sum() == 0:
        return np.zeros((3,))

    grid_size = voxelgrid.shape[0]
    x_occ, y_occ, z_occ = np.where(voxelgrid > 0.5)
    x_min = min(x_occ) / grid_size
    x_max = (max(x_occ) + 1) / grid_size
    x_center = (x_min + x_max) / 2 - 0.5

    y_min = min(y_occ) / grid_size
    y_max = (max(y_occ) + 1) / grid_size
    y_center = (y_min + y_max) / 2 - 0.5

    z_min = min(z_occ) / grid_size
    z_max = (max(z_occ) + 1) / grid_size
    z_center = (z_min + z_max) / 2 - 0.5

    return np.array((x_center, y_center, z_center))


def get_distance_to_center():
    x = np.arange(32)
    y = np.arange(32)
    xx, yy = np.meshgrid(x, y)
    xx = xx + 0.5
    yy = yy + 0.5
    d = np.sqrt(np.square(xx - int(32 / 2)) + np.square(yy - int(32 / 2)))
    return d


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def get_class(pgm):
    if pgm.dim() == 3:
        _, idx = torch.max(pgm, dim=2)
    elif pgm.dim() == 2:
        idx = pgm
    else:
        raise IndexError("dimension of pgm is wrong")
    return idx


def get_last_block(pgm):
    bsz = pgm.size(0)
    n_block = pgm.size(1)
    n_step = pgm.size(2)

    if torch.is_tensor(pgm):
        pgm = pgm.clone()
    else:
        pgm = pgm.data.clone()

    if pgm.dim() == 4:
        _, idx = torch.max(pgm, dim=3)
        idx = idx.cpu()
    elif pgm.dim() == 3:
        idx = pgm.cpu()
    else:
        raise ValueError("pgm.dim() != 2 or 3")

    max_inds = []
    for i in range(bsz):
        j = n_block - 1
        while j >= 0:
            if idx[i, j, 0] == 0:
                break
            j = j - 1

        if j == -1:
            max_inds.append(0)
        else:
            max_inds.append(j)

    return np.asarray(max_inds)


def sample_block(max_inds, include_tail=False):
    sample_inds = []
    for ind in max_inds:
        if include_tail:
            sample_inds.append(np.random.randint(0, ind + 1))
        else:
            sample_inds.append(np.random.randint(0, ind))
    return np.asarray(sample_inds)


def get_max_step_pgm(pgm):
    batch_size = pgm.size(0)

    if torch.is_tensor(pgm):
        pgm = pgm.clone()
    else:
        pgm = pgm.data.clone()

    if pgm.dim() == 3:
        pgm = pgm[:, 1:, :]
        idx = get_class(pgm).cpu()
    elif pgm.dim() == 2:
        idx = pgm[:, 1:].cpu()
    else:
        raise ValueError("pgm.dim() != 2 or 3")

    max_inds = []

    for i in range(batch_size):
        j = 0
        while j < idx.shape[1]:
            if idx[i, j] == 0:
                break
            j = j + 1
        if j == 0:
            raise ValueError("no programs for such sample")
        max_inds.append(j)

    return np.asarray(max_inds)


def get_vacancy(pgm):
    batch_size = pgm.size(0)

    if torch.is_tensor(pgm):
        pgm = pgm.clone()
    else:
        pgm = pgm.data.clone()

    if pgm.dim() == 3:
        pgm = pgm[:, 1:, :]
        idx = get_class(pgm).cpu()
    elif pgm.dim() == 2:
        idx = pgm[:, 1:].cpu()
    else:
        raise ValueError("pgm.dim() != 2 or 3")

    vac_inds = []

    for i in range(batch_size):
        j = 0
        while j < idx.shape[1]:
            if idx[i, j] == 0:
                break
            j = j + 1
        if j == idx.shape[1]:
            j = j - 1
        vac_inds.append(j)

    return np.asarray(vac_inds)


def sample_ind(max_inds, include_start=False):
    sample_inds = []
    for ind in max_inds:
        if include_start:
            sample_inds.append(np.random.randint(0, ind + 1))
        else:
            sample_inds.append(np.random.randint(0, ind))
    return np.asarray(sample_inds)


def sample_last_ind(max_inds, include_start=False):
    sample_inds = []
    for ind in max_inds:
        if include_start:
            sample_inds.append(ind)
        else:
            sample_inds.append(ind - 1)
    return np.array(sample_inds)


def decode_to_shape_new(pred_pgm, pred_param):
    batch_size = pred_pgm.size(0)

    idx = get_class(pred_pgm)

    pgm = idx.data.cpu().numpy()
    params = pred_param.data.cpu().numpy()
    params = np.round(params).astype(np.int32)

    data = np.zeros((batch_size, 32, 32, 32), dtype=np.uint8)
    for i in range(batch_size):
        for j in range(1, pgm.shape[1]):
            if pgm[i, j] == 0:
                continue
            data[i] = render_one_step_new(data[i], pgm[i, j], params[i, j])

    return data


def decode_pgm(pgm, param, loop_free=True):
    """
    decode and check one single block
    remove occasionally-happened illegal programs
    """
    flag = 1
    data_loop = []
    if pgm[0] == translate:
        if pgm[1] == translate:
            if 1 <= pgm[2] < translate:
                data_loop.append(np.hstack((pgm[0], param[0])))
                data_loop.append(np.hstack((pgm[1], param[1])))
                data_loop.append(np.hstack((pgm[2], param[2])))
                data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
                data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
            else:
                flag = 0
        elif 1 <= pgm[1] < translate:
            data_loop.append(np.hstack((pgm[0], param[0])))
            data_loop.append(np.hstack((pgm[1], param[1])))
            data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
        else:
            flag = 0
    elif pgm[0] == rotate:
        if pgm[1] == 10:
            data_loop.append(np.hstack((pgm[0], param[0])))
            data_loop.append(np.hstack((pgm[1], param[1])))
            data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
        if pgm[1] == 17:
            data_loop.append(np.hstack((pgm[0], param[0])))
            data_loop.append(np.hstack((pgm[1], param[1])))
            data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
        else:
            flag = 0
    elif 1 <= pgm[0] < translate:
        data_loop.append(np.hstack((pgm[0], param[0])))
        data_loop.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
    else:
        flag = 0

    if flag == 0:
        data_loop.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
        data_loop.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))

    data_loop = [x.tolist() for x in data_loop]
    data_loop_free = decode_loop(data_loop)
    data_loop_free = np.asarray(data_loop_free)

    if len(data_loop_free) == 0:
        data_loop_free = np.zeros((2, 8), dtype=np.int32)

    if loop_free:
        return data_loop_free
    else:
        return np.asarray(data_loop)


def decode_all(pgm, param, loop_free=False):
    """
    decode program to loop-free (or include loop)
    """
    n_block = pgm.shape[0]
    param = np.round(param).astype(np.int32)

    result = []
    for i in range(n_block):
        res = decode_pgm(pgm[i], param[i], loop_free=loop_free)
        result.append(res)
    result = np.concatenate(result, axis=0)
    return result


def execute_shape_program(pgm, param):
    """
    execute a single shape program
    """
    trace_sets = decode_all(pgm, param, loop_free=True)
    data = np.zeros((32, 32, 32), dtype=np.uint8)

    for trace in trace_sets:
        cur_pgm = trace[0]
        cur_param = trace[1:]
        data = render_one_step_new(data, cur_pgm, cur_param)

    return data


def decode_multiple_block(pgm, param):
    """
    decode and execute multiple blocks
    can run with batch style
    """
    # pgm: bsz x n_block x n_step x n_class
    # param: bsz x n_block x n_step x n_class
    bsz = pgm.size(0)
    n_block = pgm.size(1)
    data = np.zeros((bsz, 32, 32, 32), dtype=np.uint8)
    for i in range(n_block):
        if pgm.dim() == 4:
            prob_pre = torch.exp(pgm[:, i, :, :].data)
            _, it1 = torch.max(prob_pre, dim=2)
        elif pgm.dim() == 3:
            it1 = pgm[:, i, :]
        else:
            raise NotImplementedError('pgm has incorrect dimension')
        it2 = param[:, i, :, :].data.clone()
        it1 = it1.cpu().numpy()
        it2 = it2.cpu().numpy()
        data = render_block(data, it1, it2)

    return data


def decode_multiple_block_part_label(pgm, param):
    """LeoZDong addition: label each occupied voxel by their *part*
    (e.g. all 4 legs of a table is the same part)
    decode and execute multiple blocks
    Return the part index at each occupied block.
    """
    # pgm: bsz x n_block x n_step x n_class
    # param: bsz x n_block x n_step x n_class
    bsz = pgm.size(0)
    n_block = pgm.size(1)
    # NOTE: Unoccupied voxels are labeled -1 now!
    part_labels = np.ones((bsz, 32, 32, 32), dtype=np.uint8) * -1
    for i in range(n_block):
        label = i
        if pgm.dim() == 4:
            prob_pre = torch.exp(pgm[:, i, :, :].data)
            _, it1 = torch.max(prob_pre, dim=2)
        elif pgm.dim() == 3:
            it1 = pgm[:, i, :]
        else:
            raise NotImplementedError('pgm has incorrect dimension')
        it2 = param[:, i, :, :].data.clone()
        it1 = it1.cpu().numpy()
        it2 = it2.cpu().numpy()
        part_labels = render_block_part_label(part_labels, it1, it2, label)

    return part_labels


def decode_multiple_block_primitive_label(pgm, param):
    """LeoZDong addition: label each occupied voxel by their *primitive*
    (e.g. each of the 4 legs of a table is a different primitive)
    decode and execute multiple blocks
    Return the part index at each occupied block.
    """
    # pgm: bsz x n_block x n_step x n_class
    # param: bsz x n_block x n_step x n_class
    bsz = pgm.size(0)
    n_block = pgm.size(1)
    # NOTE: Unoccupied voxels are labeled -1 now!
    primitive_labels = np.ones((bsz, 32, 32, 32), dtype=np.uint8) * -1

    bsz = pgm.shape[0]
    start_label = np.zeros((bsz, ))
    for i in range(n_block):
        if pgm.dim() == 4:
            prob_pre = torch.exp(pgm[:, i, :, :].data)
            _, it1 = torch.max(prob_pre, dim=2)
        elif pgm.dim() == 3:
            it1 = pgm[:, i, :]
        else:
            raise NotImplementedError('pgm has incorrect dimension')
        it2 = param[:, i, :, :].data.clone()
        it1 = it1.cpu().numpy()
        it2 = it2.cpu().numpy()
        primitive_labels = render_block_primitive_label(
            primitive_labels, it1, it2, start_label)

    return primitive_labels


def count_blocks(pgm):
    """
    count the number of effective blocks
    """
    # pgm: bsz x n_block x n_step x n_class
    pgm = pgm.data.clone().cpu()
    bsz = pgm.size(0)
    n_blocks = []
    n_for = []
    for i in range(bsz):
        prob = torch.exp(pgm[i, :, :, :])
        _, it = torch.max(prob, dim=2)
        v = it[:, 0].numpy()
        n_blocks.append((v > 0).sum())
        n_for.append((v == translate).sum() + (v == rotate).sum())

    return np.asarray(n_blocks), np.asarray(n_for)


def render_new(data, pgms, params):
    """
    render one step for a batch
    """
    batch_size = data.shape[0]
    params = np.round(params).astype(np.int32)

    for i in range(batch_size):
        data[i] = render_one_step_new(data[i], pgms[i], params[i])

    return data


def render_block(data, pgm, param):
    """
    render one single block
    """
    param = np.round(param).astype(np.int32)
    bsz = data.shape[0]
    for i in range(bsz):
        loop_free = decode_pgm(pgm[i], param[i])
        cur_pgm = loop_free[:, 0]
        cur_param = loop_free[:, 1:]
        for j in range(len(cur_pgm)):
            data[i] = render_one_step_new(data[i], cur_pgm[j], cur_param[j])

    return data


def render_block_part_label(data, pgm, param, label):
    """LeoZDong addition: return part label at occupied voxels instead of 0/1.
    render one single block
    """
    param = np.round(param).astype(np.int32)
    bsz = data.shape[0]
    for i in range(bsz):
        loop_free = decode_pgm(pgm[i], param[i])
        cur_pgm = loop_free[:, 0]
        cur_param = loop_free[:, 1:]
        for j in range(len(cur_pgm)):
            data[i] = render_one_step_new(data[i], cur_pgm[j], cur_param[j], label=label)

    return data


def render_block_primitive_label(data, pgm, param, start_label):
    """LeoZDong addition: return primitive label at occupied voxels instead of 0/1.
    render one single block
    """
    param = np.round(param).astype(np.int32)
    bsz = data.shape[0]
    for i in range(bsz):
        loop_free = decode_pgm(pgm[i], param[i])
        cur_pgm = loop_free[:, 0]
        cur_param = loop_free[:, 1:]
        for j in range(len(cur_pgm)):
            data[i] = render_one_step_new(data[i],
                                          cur_pgm[j],
                                          cur_param[j],
                                          label=start_label[i])
            start_label[i] += 1

    return data


def render_one_step_new(data, pgm, param, label=None):
    """LeoZDong edit: take label as parameter to each draw command.
    render one step
    """
    if pgm == 0:
        pass
    elif pgm == 1:
        data = draw_vertical_leg_new(data, param[0], param[1], param[2], param[3], param[4], param[5], label=label)[0]
    elif pgm == 2:
        data = draw_rectangle_top_new(data, param[0], param[1], param[2], param[3], param[4], param[5], label=label)[0]
    elif pgm == 3:
        data = draw_square_top_new(data, param[0], param[1], param[2], param[3], param[4], label=label)[0]
    elif pgm == 4:
        data = draw_circle_top_new(data, param[0], param[1], param[2], param[3], param[4], label=label)[0]
    elif pgm == 5:
        data = draw_middle_rect_layer_new(data, param[0], param[1], param[2], param[3], param[4], param[5], label=label)[0]
    elif pgm == 6:
        data = draw_circle_support_new(data, param[0], param[1], param[2], param[3], param[4], label=label)[0]
    elif pgm == 7:
        data = draw_square_support_new(data, param[0], param[1], param[2], param[3], param[4], label=label)[0]
    elif pgm == 8:
        data = draw_circle_base_new(data, param[0], param[1], param[2], param[3], param[4], label=label)[0]
    elif pgm == 9:
        data = draw_square_base_new(data, param[0], param[1], param[2], param[3], param[4], label=label)[0]
    elif pgm == 10:
        data = draw_cross_base_new(data, param[0], param[1], param[2], param[3], param[4], param[5], label=label)[0]
    elif pgm == 11:
        data = draw_sideboard_new(data, param[0], param[1], param[2], param[3], param[4], param[5], label=label)[0]
    elif pgm == 12:
        data = draw_horizontal_bar_new(data, param[0], param[1], param[2], param[3], param[4], param[5], label=label)[0]
    elif pgm == 13:
        data = draw_vertboard_new(data, param[0], param[1], param[2], param[3], param[4], param[5], label=label)[0]
    elif pgm == 14:
        data = draw_locker_new(data, param[0], param[1], param[2], param[3], param[4], param[5], label=label)[0]
    elif pgm == 15:
        data = draw_tilt_back_new(data, param[0], param[1], param[2], param[3], param[4], param[5], param[6], label=label)[0]
    elif pgm == 16:
        data = draw_chair_beam_new(data, param[0], param[1], param[2], param[3], param[4], param[5], label=label)[0]
    elif pgm == 17:
        data = draw_line_new(data, param[0], param[1], param[2], param[3], param[4], param[5], param[6], label=label)[0]
    elif pgm == 18:
        data = draw_back_support_new(data, param[0], param[1], param[2], param[3], param[4], param[5], label=label)[0]
    else:
        raise RuntimeError("program id is out of range, pgm={}".format(pgm))

    return data


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# LeoZDong addition: 6 high contrast colors
HIGH_CONTRAST = np.zeros((10, 3))
HIGH_CONTRAST[0] = [137, 49, 239]
HIGH_CONTRAST[1] = [242, 202, 25]
HIGH_CONTRAST[2] = [255, 0, 189]
HIGH_CONTRAST[3] = [0, 87, 233]
HIGH_CONTRAST[4] = [135, 233, 17]
HIGH_CONTRAST[5] = [225, 24, 69]
HIGH_CONTRAST[6] = [0, 0, 0]
HIGH_CONTRAST[7] = [0, 255, 0]
HIGH_CONTRAST[8] = [0, 0, 255]
HIGH_CONTRAST[9] = [225, 0, 0]
HIGH_CONTRAST /= 255