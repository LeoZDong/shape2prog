from __future__ import print_function

import os

from torch.utils.data import Dataset
import numpy as np
import h5py
from programs.label_config import num_params, max_param
from programs.loop_gen import translate, rotate, end
import binvox_rw
import yaml

class PartPrimitive(Dataset):
    """
    dataset for (part, block program) pairs
    """
    def __init__(self, file_path):
        f = h5py.File(file_path, 'r')
        self.data = np.array(f['data'])
        self.labels = np.array(f['label'])

        assert self.data.shape[0] == self.labels.shape[0]

        self.num = self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index, :, 0]
        param = self.labels[index, :, 1:]

        data = data.astype(np.int64)
        label = label.astype(np.int64)
        param = param.astype(np.float32)

        return data, label, param

    def __len__(self):
        return self.num


class Synthesis3D(Dataset):
    """
    dataset for (shape, program) pairs
    """
    def __init__(self, file_path, n_block=6, n_step=3, w1=1, w2=1):
        f = h5py.File(file_path, 'r')
        self.data = np.array(f['data'])
        self.labels = np.array(f['programs'])
        self.n_block = n_block
        self.n_step = n_step
        self.max_block = 0
        self.pgm_weight = w1
        self.param_weight = w2

        assert self.data.shape[0] == self.labels.shape[0]

        self.num = self.data.shape[0]

        self.pgms = np.zeros((self.num, self.n_block, self.n_step), dtype=np.int32)
        self.pgm_masks = np.zeros((self.num, self.n_block, self.n_step), dtype=np.float32)
        self.params = np.zeros((self.num, self.n_block, self.n_step, max_param-1), dtype=np.float32)
        self.param_masks = np.zeros((self.num, self.n_block, self.n_step, max_param-1), dtype=np.float32)

        for i in range(self.num):
            pgm, param, pgm_mask, param_mask = self.process_label(self.labels[i])
            self.pgms[i] = pgm
            self.pgm_masks[i] = pgm_mask
            self.params[i] = param
            self.param_masks[i] = param_mask

    def __getitem__(self, index):
        data = np.copy(self.data[index])

        pgm = self.pgms[index]
        pgm_mask = self.pgm_masks[index]
        param = self.params[index]
        param_mask = self.param_masks[index]

        data = data.astype(np.float32)
        pgm = pgm.astype(np.int64)

        return data, pgm, pgm_mask, param, param_mask

    def __len__(self):
        return self.num

    def process_label(self, label):
        pgm = np.zeros((self.n_block, self.n_step), dtype=np.int32)
        param = np.zeros((self.n_block, self.n_step, max_param - 1), dtype=np.float32)
        pgm_mask = 0.1 * np.ones((self.n_block, self.n_step), dtype=np.float32)
        param_mask = 0.1 * np.ones((self.n_block, self.n_step, max_param - 1), dtype=np.float32)

        max_step = label.shape[0]

        pgm_weight = self.pgm_weight
        param_weight = self.param_weight

        i = 0
        j = 0
        while j < max_step:
            if label[j, 0] == translate:
                if label[j+1, 0] == translate:
                    # pgm
                    pgm[i, 0] = translate
                    pgm[i, 1] = translate
                    pgm[i, 2] = label[j+2, 0]
                    pgm_mask[i, :3] = pgm_weight
                    # param
                    param[i, :3] = label[j:j+3, 1:]
                    param_mask[i, :2, :num_params[translate]] = param_weight
                    param_mask[i, 2, :num_params[pgm[i, 2]]] = param_weight
                    j = j + 5
                    i = i + 1
                else:
                    # pgm
                    pgm[i, 0] = translate
                    pgm[i, 1] = label[j+1, 0]
                    pgm_mask[i, :2] = pgm_weight
                    # param
                    param[i, :2] = label[j:j+2, 1:]
                    param_mask[i, 0, :num_params[translate]] = param_weight
                    param_mask[i, 1, :num_params[pgm[i, 1]]] = param_weight
                    j = j + 3
                    i = i + 1
            elif label[j, 0] == rotate:
                # pgm
                pgm[i, 0] = rotate
                pgm[i, 1] = label[j+1, 0]
                pgm_mask[i, :2] = pgm_weight
                # param
                param[i, :2] = label[j:j+2, 1:]
                param_mask[i, 0, :num_params[rotate]] = param_weight
                param_mask[i, 1, :num_params[pgm[i, 1]]] = param_weight
                j = j + 3
                i = i + 1
            elif label[j, 0] == end:
                j = j + 1
            elif label[j, 0] > 0:
                # pgm
                pgm[i, 0] = label[j, 0]
                pgm_mask[i, 0] = pgm_weight
                # param
                param[i, 0] = label[j, 1:]
                param_mask[i, 0, :num_params[pgm[i, 0]]] = param_weight
                j = j + 1
                i = i + 1
            else:
                break

            if i == self.n_block:
                print(label)

        if i > self.max_block:
            self.max_block = i

        return pgm, param, pgm_mask, param_mask


class ShapeNet3D(Dataset):
    """
    dataset for ShapeNet
    """
    def __init__(self, file_path):
        super(ShapeNet3D, self).__init__()

        f = h5py.File(file_path, "r")
        self.data = np.array(f['data'])
        self.num = self.data.shape[0]

    def __getitem__(self, index):
        data = np.copy(self.data[index, ...])
        data = data.astype(np.float32)

        return data

    def __len__(self):
        return self.num

class Shapes3dDataset(Dataset):
    """3D Shapes dataset class.
    Adopted from: https://github.com/autonomousvision/occupancy_networks
    """

    def __init__(self, dataset_folder, fields, split=None, categories=None, i=-1):
        """Initialization of the the 3D shape dataset.
        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
        """
        self.i = i  # For debugging

        # Attributes
        self.dataset_folder = dataset_folder
        self.split = split
        self.fields = fields

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            }

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models in split
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            }

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                print('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')

            self.models += [
                {'category': c, 'model': m}
                for m in models_c
            ]

        # TEMP: rotate about Y axis by 90 degrees
        # self.rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)


    def __len__(self):
        """Returns the length of the dataset.
        """
        if self.i != -1:
            return 2
        else:
            return len(self.models)

    def __getitem__(self, idx):
        """Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        """
        if self.i != -1:
            idx = 0

        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}

        for field_name, field in self.fields.items():
            try:
                field_data = field.load(model_path, idx, c_idx)
            except Exception:
                if self.no_except:
                    print(
                        'Error occured when loading field %s of model %s'
                        % (field_name, model)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        # My Edit: Also return model ID
        data['id'] = model
        data['c'] = category

        # Comment below when running train_debug.py
        data = self.change_data_format(data)

        return data

    def change_data_format(self, data):
        """Change data output format."""
        voxels = data['voxels']
        voxels = np.swapaxes(voxels, 0, 1)
        voxels = np.flip(voxels, 1)
        
        # This step is necessary because torch dataloader does not work with negative stride
        voxels = voxels.copy()

        return voxels


    def get_model_dict(self, idx):
        return self.models[idx]


class VoxelsField(object):
    ''' Voxel field class.
    It provides the class used for voxel-based data.
    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
