from __future__ import print_function

import os
import argparse
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import h5py
from torch.utils.data import DataLoader
from torch.autograd import Variable

# from visualization.util_vtk import visualization
from dataset import ShapeNet3D, Shapes3dDataset, VoxelsField, PointCloudField, PointsField
from model import BlockOuterNet
from criterion import BatchIoU
from misc import *
from interpreter import Interpreter
from programs.loop_gen import translate, rotate, end

import socket
import pytorch3d
from pytorch3d.io import save_ply
# import open3d as o3d
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds
import matplotlib
import matplotlib.cm


def parse_argument():

    parser = argparse.ArgumentParser(description="testing the program generator")

    parser.add_argument('--cls', type=str, default='chair')
    parser.add_argument('--data', type=str, default='./data/chair_testing.h5',
                        help='path to the testing data')
    parser.add_argument('--data_folder',
                        type=str,
                        default='data',
                        help='directory to data')

    parser.add_argument('--scale_down', action='store_true', default=False)
    parser.add_argument('--segpoints', action='store_true', default=False)

    parser.add_argument('--save_path', type=str, default='./output/chair/',
                        help='path to save the output results')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--info_interval', type=int, default=10, help='freq for printing info')

    parser.add_argument('--save_prog', action='store_true', help='save programs to text file')
    parser.add_argument('--save_img', action='store_true', help='render reconstructed shapes to images')
    parser.add_argument('--save_shapes', action='store_true')

    parser.add_argument('--num_render', type=int, default=10, help='how many samples to be rendered')

    parser.add_argument('--model_name', type=str, default='GA')

    opt = parser.parse_args()

    opt.mesh_save_path = os.path.join(opt.save_path, opt.model_name, 'meshes')
    opt.prog_save_path = os.path.join(opt.save_path, opt.model_name, 'programs')
    opt.segpoints_save_path = os.path.join(opt.save_path, 'segpoints')

    opt.model = os.path.join('model', f'ckpts_{opt.model_name}_{opt.cls}',
                             f'program_generator_{opt.model_name}_{opt.cls}.t7')

    opt.is_cuda = torch.cuda.is_available()

    return opt


def test_on_shapenet_data(epoch, test_loader, model, opt, gen_shape=False,
                          gen_segpoints_labels=False):

    model.eval()
    generated_shapes = []  # voxels
    original_shapes = []
    gen_pgms = []
    gen_params = []

    generated_meshes = []  # aligned meshes
    ids = []  # ShapeNet ids of the generated shapes
    original_pcs = []
    ious = []

    # For segmentation evaluation
    vox_part_labels = []  # for visualization debugging only
    vox_primitive_labels = []  # for visualization debugging only
    segpoints_part_labels = []
    segpoints_primitive_labels = []
    segpoints_n = []

    for idx, data in enumerate(test_loader):
        start = time.time()
        shapes = data['voxels']
        shapes = Variable(torch.unsqueeze(shapes, 1), requires_grad=False).cuda()
        out = model.decode(shapes)

        # Always generate programs
        _, save_pgms = torch.max(out[0].detach(), dim=3)
        save_pgms = save_pgms.cpu().numpy()
        save_params = out[1].detach().cpu().numpy()
        gen_pgms.append(save_pgms)
        gen_params.append(save_params)

        if opt.is_cuda:
            torch.cuda.synchronize()
        end = time.time()

        # Get important information for alignment
        pointcloud = data['pointcloud']
        target_sizes, target_centers = get_sizes_and_centers(pointcloud)
        vox_side_len = 24 if opt.scale_down else 32

        # Generate segmentation points' labels
        if gen_segpoints_labels:
            # IMPORTANT NOTE: For some reason, running part and primitive together
            # gives wrong results probably for some in-place assignment I messed up.
            # For now, just run them separately.

            # Part labels:
            # pred_part_labels = decode_multiple_block_part_label(out[0], out[1])
            # vox_part_labels.append(pred_part_labels)
            # points_part_labels = get_segpoints_labels_batch(
            #     data['segpoints'].numpy(),
            #     pred_part_labels,
            #     target_centers.numpy(),
            #     target_sizes.numpy(),
            #     vox_side_len)
            # segpoints_part_labels.append(points_part_labels)

            # Primitive labels:
            pred_primitive_labels = decode_multiple_block_primitive_label(out[0], out[1])
            vox_primitive_labels.append(pred_primitive_labels)
            points_primitive_labels = get_segpoints_labels_batch(
                data['segpoints'].numpy(), pred_primitive_labels,
                target_centers.numpy(), target_sizes.numpy(), vox_side_len)
            segpoints_primitive_labels.append(points_primitive_labels)

            segpoints_n.append(data['segpoints_n'].numpy())

        # Generate output shapes (both voxels and aligned meshes)
        if gen_shape:
            pred_voxels = decode_multiple_block(out[0], out[1])
            generated_shapes.append(pred_voxels)
            original_shapes.append(data['voxels'].clone().cpu().numpy())
            original_pcs.append(pointcloud)

            # Generate aligned meshes
            aligned_meshes = voxel_to_aligned_mesh(pred_voxels, target_sizes, target_centers)
            verts_list = aligned_meshes.verts_list()
            faces_list = aligned_meshes.faces_list()

            for i in range(len(verts_list)):
                verts, faces = verts_list[i], faces_list[i]
                generated_meshes.append((verts, faces))

            # Compute voxel to mesh iou
            ious_batch = vox_mesh_iou_batch(pred_voxels, target_sizes.numpy(),
                                            target_centers.numpy(),
                                            data['points'].numpy(),
                                            data['points.occ'].numpy(),
                                            vox_side_len)

            # DEBUG: use original voxels instead!
            # ious_batch = vox_mesh_iou_batch(data['voxels'].clone().cpu().numpy(),
            #                                 target_sizes.numpy(),
            #                                 target_centers.numpy(),
            #                                 data['points'].numpy(),
            #                                 data['points.occ'].numpy(),
            #                                 vox_side_len,
            #                                 pc=pointcloud.numpy())
            ious += ious_batch

        ids += data['id']


        if idx % opt.info_interval == 0:
            print("Test: epoch {} batch {}/{}, time={:.3f}".format(epoch, idx, len(test_loader), end - start))

    if gen_shape:
        original_pcs = torch.cat(original_pcs, 0)
        generated_shapes = np.concatenate(generated_shapes, axis=0)
        original_shapes = np.concatenate(original_shapes, axis=0)
        gen_pgms = np.concatenate(gen_pgms, axis=0)
        gen_params = np.concatenate(gen_params, axis=0)

    if gen_segpoints_labels:
        # vox_part_labels = np.concatenate(vox_part_labels, 0)
        # segpoints_part_labels = np.concatenate(segpoints_part_labels, 0)
        vox_primitive_labels = np.concatenate(vox_primitive_labels, 0)
        segpoints_primitive_labels = np.concatenate(segpoints_primitive_labels, 0)
        segpoints_n = np.concatenate(segpoints_n, 0)

    return {
        'ori_shapes': original_shapes,
        'gen_shapes': generated_shapes,
        'gen_meshes': generated_meshes,
        'pgms': gen_pgms,
        'params': gen_pgms,
        'ids': ids,
        'ori_pcs': original_pcs,
        'ious': ious,
        'vox_part_labels': vox_part_labels,
        'vox_primitive_labels': vox_primitive_labels,
        'segpoints_part_labels': segpoints_part_labels,
        'segpoints_primitive_labels': segpoints_primitive_labels,
        'segpoints_n': segpoints_n
    }


def run():
    opt = parse_argument()

    if not os.path.isdir(opt.prog_save_path):
        os.makedirs(opt.prog_save_path)
    if not os.path.isdir(opt.mesh_save_path):
        os.makedirs(opt.mesh_save_path)
    if not os.path.isdir(opt.segpoints_save_path):
        os.makedirs(opt.segpoints_save_path)

    print('========= arguments =========')
    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
    print('========= arguments =========')

    # data loader
    # test_set = ShapeNet3D(opt.data)
    voxel_field = VoxelsField('model.binvox')
    pc_field = PointCloudField('pointcloud.npz')
    pt_field = PointsField('points.npz', True)

    categ_to_id = {'chair': '03001627', 'table': '04379243'}
    categories = [categ_to_id[opt.cls]]
    test_set = Shapes3dDataset(opt.data_folder, {
        'voxels': voxel_field,
        'pointcloud': pc_field,
        'points': pt_field
    },
                               split='train_parts' if opt.segpoints else 'test',
                               categories=categories,
                               scale_down=opt.scale_down,
                               segpoints=opt.segpoints)
    print("Test set size:", len(test_set))
    test_set[0]

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )

    # model
    ckpt = torch.load(opt.model)
    model = BlockOuterNet(ckpt['opt'])
    model.load_state_dict(ckpt['model'])
    if opt.is_cuda:
        model = model.cuda()
        cudnn.benchmark = True


    # test the model and evaluate the IoU
    test_out = test_on_shapenet_data(
        epoch=0,
        test_loader=test_loader,
        model=model,
        opt=opt,
        gen_shape=False,
        gen_segpoints_labels=opt.segpoints)

    ori_shapes = test_out['ori_shapes']
    gen_shapes = test_out['gen_shapes']
    pgms = test_out['pgms']
    params = test_out['params']
    gen_meshes = test_out['gen_meshes']
    ids = test_out['ids']
    ori_pcs = test_out['ori_pcs']
    ious = test_out['ious']
    vox_part_labels = test_out['vox_part_labels']
    vox_primitive_labels = test_out['vox_primitive_labels']
    segpoints_part_labels = test_out['segpoints_part_labels']
    segpoints_primitive_labels = test_out['segpoints_primitive_labels']
    segpoints_n = test_out['segpoints_n']

    if opt.save_shapes:
        IoU = BatchIoU(ori_shapes, gen_shapes)
        print("Mean voxel-voxel IoU: {:.3f}".format(IoU.mean()))
        print("Mean mesh-voxel IoU: ", sum(ious) / len(ious))

    # execute the generated program to generate the reconstructed shapes
    # for double-check purpose, can be disabled
    # num_shapes = gen_shapes.shape[0]
    # res = []
    # for i in range(num_shapes):
    #     data = execute_shape_program(pgms[i], params[i])
    #     res.append(data.reshape((1, 32, 32, 32)))
    # res = np.concatenate(res, axis=0)
    # IoU_2 = BatchIoU(ori_shapes, res)

    # assert abs(IoU.mean() - IoU_2.mean()) < 0.1, 'IoUs are not matched'

    # save results
    # save_file = os.path.join(opt.save_path, 'shapes.h5')
    # f = h5py.File(save_file, 'w')
    # f['data'] = gen_shapes
    # f['pgms'] = pgms
    # f['params'] = params
    # f.close()

    # Save segpoints labels
    if opt.segpoints:
        size = len(segpoints_primitive_labels)
        for i in range(size):
            id = ids[i]
            save_dir = os.path.join(opt.segpoints_save_path, opt.cls, 'train', id)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            # Remove padding
            n_points = segpoints_n[i]
            # part_labels = segpoints_part_labels[i][:n_points]
            # np.save(os.path.join(save_dir, 'segpoints_part_labels.npy'),
            #         part_labels)

            primitive_labels = segpoints_primitive_labels[i][:n_points]
            np.save(os.path.join(save_dir, 'segpoints_primitive_labels.npy'),
                    primitive_labels)

            # Visualize for debugging
            # c = HIGH_CONTRAST[part_labels.astype(int) % 10]
            # data = test_set[i]
            # segpoints = data['segpoints'][:n_points]
            # pointcloud = data['pointcloud']

            # # pcd = o3d.geometry.PointCloud()
            # # pcd.points = o3d.utility.Vector3dVector(segpoints)
            # # pcd.colors = o3d.utility.Vector3dVector(c)
            # # o3d.io.write_point_cloud(os.path.join(save_dir, 'segpoints_prim.ply'), pcd)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(segpoints)
            # pcd.colors = o3d.utility.Vector3dVector(c)
            # o3d.io.write_point_cloud(
            #     os.path.join(save_dir, 'segpoints_prim.ply'), pcd)

            # pcd2 = o3d.geometry.PointCloud()
            # pcd2.points = o3d.utility.Vector3dVector(pointcloud)
            # o3d.io.write_point_cloud(os.path.join(save_dir, 'pointcloud.ply'), pcd2)

            # verts, faces = gen_meshes[i]
            # save_file = os.path.join(save_dir, 'mesh.ply')
            # save_ply(save_file, verts, faces)

    # Save meshes
    if opt.save_shapes:
        for i in range(min(len(gen_meshes), opt.num_render)):
            verts, faces = gen_meshes[i]
            id = ids[i]

            save_dir = os.path.join(opt.mesh_save_path, opt.cls, id)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            save_file = os.path.join(save_dir, 'mesh.ply')
            save_ply(save_file, verts, faces)

            # Also save ground-truth pointcloud
            # Save with Open3D:
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(gt_pc)
            # o3d.io.write_point_cloud(save_file, pcd)
            # IMPORTANT NODE: I need to do `gt_pc = torch.stack((z, y, x), 1)` to
            # get the pointcloud in same orientation as the voxel, because they are
            # not aligned in the occnet dataset. Need to swap mesh accordingly.

            # Save with pytorch3d:
            # gt_pc = ori_pcs[i]
            # save_file = os.path.join(save_dir, 'pointcloud.ply')
            # p3d_pc = Pointclouds([gt_pc])
            # io = IO()
            # io.save_pointcloud(p3d_pc, path=save_file)

    # Interpreting programs to understandable program strings
    if opt.save_prog:
        num_parts = []
        if not os.path.isdir(os.path.join(opt.prog_save_path, opt.cls)):
            os.makedirs(os.path.join(opt.prog_save_path, opt.cls))
        interpreter = Interpreter(translate, rotate, end)
        num_programs = gen_shapes.shape[0]
        for i in range(min(num_programs, opt.num_render)):
            program, n_parts = interpreter.interpret(pgms[i], params[i], return_n_parts=True)
            num_parts.append(n_parts)
            save_file = os.path.join(opt.prog_save_path, opt.cls,
                                     '{}.txt'.format(ids[i]))
            with open(save_file, 'w') as out:
                out.write(program)

        print("Average number of part (blocks):", sum(num_parts) / len(num_parts))


if __name__ == '__main__':
    run()
