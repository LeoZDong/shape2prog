from __future__ import print_function

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import VoxelsField, Shapes3dDataset, ShapeNet3D
from model import BlockOuterNet, RenderNet
from criterion import BatchIoU
from misc import clip_gradient, decode_multiple_block, scale_voxels
from options import options_guided_adaptation
from plot_voxels import plot_voxels

def train(epoch, train_loader, generator, executor, soft, criterion, optimizer, opt):
    """
    one epoch guided adaptation
    """
    generator.train()

    # set executor as train, but actually does not update parameters
    # otherwise cannot bp through LSTM
    executor.train()

    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    executor.apply(set_bn_eval)

    for idx, data in enumerate(train_loader):
        start = time.time()

        optimizer.zero_grad()
        generator.zero_grad()
        executor.zero_grad()

        shapes = data
        raw_shapes = data

        shapes = torch.unsqueeze(shapes, 1)
        if opt.is_cuda:
            shapes = shapes.cuda()

        pgms, params = generator.decode(shapes)

        # truly rendered shapes
        rendered_shapes = decode_multiple_block(pgms, params)
        IoU2 = BatchIoU(rendered_shapes, raw_shapes.clone().numpy())

        # neurally rendered shapes
        pgms = torch.exp(pgms)
        bsz, n_block, n_step, n_vocab = pgms.shape
        pgm_vector = pgms.view(bsz * n_block, n_step, n_vocab)
        bsz, n_block, n_step, n_param = params.shape
        param_vector = params.view(bsz * n_block, n_step, n_param)
        index = (n_step - 1) * torch.ones(bsz * n_block).long()
        if opt.is_cuda:
            index = index.cuda()

        pred = executor(pgm_vector, param_vector, index)
        pred = soft(pred)
        pred = pred[:, 1, :, :, :]
        pred = pred.contiguous().view(bsz, n_block, 32, 32, 32)

        rec, _ = torch.max(pred[:, :, :, :, :], dim=1)
        rec1 = rec
        rec1.unsqueeze_(1)
        rec0 = 1 - rec1
        rec_all = torch.cat((rec0, rec1), dim=1)
        rec_all = torch.log(rec_all + 1e-10)
        loss = criterion(rec_all, shapes.detach().squeeze(1).long())

        loss.backward()
        clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()

        reconstruction = rec.data.cpu().numpy()
        reconstruction = np.squeeze(reconstruction, 1)
        reconstruction = reconstruction > 0.5
        reconstruction = reconstruction.astype(np.uint8)
        raw_shapes = raw_shapes.clone().numpy()
        IoU1 = BatchIoU(reconstruction, raw_shapes)

        if opt.is_cuda:
            torch.cuda.synchronize()

        end = time.time()

        if idx % opt.info_interval == 0:
            print("Train: epoch {} batch {}/{}, loss = {:.3f}, IoU1 = {:.3f}, IoU2 = {:.3f}, time = {:.3f}"
                  .format(epoch, idx, len(train_loader), loss.item(), IoU1.mean(), IoU2.mean(), end - start))
            sys.stdout.flush()


def validate(epoch, val_loader, generator, opt, gen_shape=False):
    """
    evaluate program generator, in terms of IoU
    """
    generator.eval()
    generated_shapes = []
    original_shapes = []

    for idx, data in enumerate(val_loader):
        start = time.time()
        shapes = data
        shapes = torch.unsqueeze(shapes, 1)

        if opt.is_cuda:
            shapes = shapes.cuda()

        out = generator.decode(shapes)

        if opt.is_cuda:
            torch.cuda.synchronize()

        end = time.time()

        if gen_shape:
            generated_shapes.append(decode_multiple_block(out[0], out[1]))
            original_shapes.append(data.clone().numpy())

        if idx % opt.info_interval == 0:
            print("Test: epoch {} batch {}/{}, time={:.3f}"
                  .format(epoch, idx, len(val_loader), end - start))

    if gen_shape:
        generated_shapes = np.concatenate(generated_shapes, axis=0)
        original_shapes = np.concatenate(original_shapes, axis=0)

    return generated_shapes, original_shapes


def run():
    # get options
    opt = options_guided_adaptation.parse()

    print('===== arguments: guided adaptation =====')
    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
    print('===== arguments: guided adaptation =====')

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # Visualize synthetic shapes
    # syn_set = ShapeNet3D('data/chair_testing.h5')
    # syn_voxels = syn_set[0]
    # plot_voxels(syn_voxels, 'plots', 'syn_voxels.png')

    #### Switch to custom ShapeNet data loader ####
    voxel_field = VoxelsField('model.binvox')
    categ_to_id = {'chair': '03001627', 'table': '04379243'}
    categories = [categ_to_id[opt.cls]]

    if opt.scale_down:
        print("Scale down longest side from 32 to 24!")

    train_set = Shapes3dDataset(opt.data_folder,
                                {'voxels': voxel_field},
                                split='train',
                                categories=categories,
                                scale_down=opt.scale_down)
    print("Train set size:", len(train_set))
    # Visualize shapenet shapes (to make sure rotation / axes are aligned with synthetic)
    # shapenet_voxels = train_set[0]
    # plot_voxels(shapenet_voxels, 'plots', 'shapenet_voxels.png')
    # plot_voxels(scale_voxels(shapenet_voxels, 0.75), 'plots', 'shapenet_voxels_rescale.png')

    val_set = Shapes3dDataset(opt.data_folder, {'voxels': voxel_field},
                              split='val',
                              categories=categories,
                              scale_down=opt.scale_down)
    print("Val set size:", len(val_set))

    #### Use original ShapeNet data loader ####
    # train_set = ShapeNet3D(opt.train_file)
    # print("Train set size:", len(train_set))
    # val_set = ShapeNet3D(opt.val_file)
    # print("Val set size:", len(val_set))

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )

    # load program generator
    ckpt_p_gen = torch.load(opt.p_gen_path)
    # ckpt_p_gen = torch.load('model/program_generator_GA_chair.t7')
    generator = BlockOuterNet(ckpt_p_gen['opt'])
    generator.load_state_dict(ckpt_p_gen['model'])

    # load program executor
    ckpt_p_exe = torch.load(opt.p_exe_path)
    executor = RenderNet(ckpt_p_exe['opt'])
    executor.load_state_dict(ckpt_p_exe['model'])

    # build loss functions
    soft = nn.Softmax(dim=1)
    criterion = nn.NLLLoss(weight=torch.Tensor([1, 1]))

    if opt.is_cuda:
        generator = generator.cuda()
        executor = executor.cuda()
        soft = soft.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    optimizer = optim.Adam(generator.parameters(),
                           lr=opt.learning_rate,
                           betas=(opt.beta1, opt.beta2),
                           weight_decay=opt.weight_decay)

    print("###################")
    print("testing")
    gen_shapes, ori_shapes = validate(0, val_loader, generator, opt,
                                      gen_shape=True)
    IoU = BatchIoU(ori_shapes, gen_shapes)
    print("iou: ", IoU.mean())
    # exit()

    best_iou = 0

    for epoch in range(1, opt.epochs+1):
        print("###################")
        print("adaptation")
        train(epoch, train_loader, generator, executor, soft, criterion, optimizer, opt)
        print("###################")
        print("testing")
        gen_shapes, ori_shapes = validate(epoch, val_loader, generator, opt,
                                          gen_shape=True)
        IoU = BatchIoU(ori_shapes, gen_shapes)
        print("iou: ", IoU.mean())

        if epoch % opt.save_interval == 0:
            print('Saving...')
            state = {
                'opt': ckpt_p_gen['opt'],
                'model': generator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.t7'.format(epoch=epoch))
            torch.save(state, save_file)

        if IoU.mean() >= best_iou:
            print('Saving best model')
            state = {
                'opt': ckpt_p_gen['opt'],
                'model': generator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(opt.save_folder, 'program_generator_GA_{}.t7'.format(opt.cls))
            torch.save(state, save_file)
            best_iou = IoU.mean()


if __name__ == '__main__':
    run()
