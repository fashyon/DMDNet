import os.path
from os.path import join
from data.transforms import to_tensor, ReflectionSythesis_1
from PIL import Image
import random
import torch
import math
import torchvision.transforms.functional as F
import data.torchdata as torchdata
import numpy as np
from torch.utils.data import Dataset,DataLoader
from scipy.signal import convolve2d
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, fns=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if fns is None:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    else:
        for fname in fns:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                images.append(path)

    return images


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

def get_params(img, output_size):
    w, h = img.size
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw

def paired_data_transforms(img_1, img_2,patchsize):
    img_size = min(patchsize)
    target_size = int(random.randint(img_size, img_size*2) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)

    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img_1 = F.rotate(img_1, angle)
        img_2 = F.rotate(img_2, angle)

    i, j, h, w = get_params(img_1, patchsize)
    img_1 = F.crop(img_1, i, j, h, w)
    img_2 = F.crop(img_2, i, j, h, w)

    img_1 = __scale_width(img_1, img_size)
    img_2 = __scale_width(img_2, img_size)

    return img_1, img_2

def triplet_data_transforms(img_1, img_2, img_3,patchsize):
    img_size = min(patchsize)
    target_size = int(random.randint(img_size, img_size*2) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
        img_3 = __scale_height(img_3, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)
        img_3 = __scale_width(img_3, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)
        img_3 = F.hflip(img_3)

    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img_1 = F.rotate(img_1, angle)
        img_2 = F.rotate(img_2, angle)
        img_3 = F.rotate(img_3, angle)

    i, j, h, w = get_params(img_1, patchsize)
    img_1 = F.crop(img_1, i, j, h, w)
    img_2 = F.crop(img_2, i, j, h, w)
    img_3 = F.crop(img_3, i, j, h, w)

    img_1 = __scale_width(img_1, img_size)
    img_2 = __scale_width(img_2, img_size)
    img_3 = __scale_width(img_3, img_size)

    return img_1, img_2,img_3

BaseDataset = torchdata.Dataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()

class ReflectionSynthesis(object):
    def __init__(self):
        # Kernel Size of the Gaussian Blurry
        self.kernel_sizes = [5, 7, 9, 11]
        self.kernel_probs = [0.1, 0.2, 0.3, 0.4]

        # Sigma of the Gaussian Blurry
        self.sigma_range = [2, 5]
        self.alpha_range = [0.8, 1.0]
        self.beta_range = [0.4, 1.0]

    def __call__(self, T_, R_):
        T_ = np.asarray(T_, np.float32) / 255.
        R_ = np.asarray(R_, np.float32) / 255.

        kernel_size = np.random.choice(self.kernel_sizes, p=self.kernel_probs)
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel2d = np.dot(kernel, kernel.T)
        for i in range(3):
            R_[..., i] = convolve2d(R_[..., i], kernel2d, mode='same')

        a = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
        b = np.random.uniform(self.beta_range[0], self.beta_range[1])
        T, R = a * T_, b * R_
        I = T + R - T * R
        return T_, R_, I

class CEILDataset(BaseDataset):
    def __init__(self, datadir,fns=None, size=None,patchsize=(None,None),enable_transforms=True,shuffle=False):
        super(CEILDataset, self).__init__()
        self.size = size
        self.patchsize = patchsize
        self.datadir = datadir
        self.enable_transforms = enable_transforms
        self.syn_model = ReflectionSynthesis()
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths = sorted(make_dataset(datadir, fns), key=sortkey)
        if size is not None:
            self.paths = self.paths[:size]
        self.reset(shuffle=shuffle)

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.paths)
        num_images = len(self.paths) // 2
        self.T_paths = self.paths[:num_images]
        self.R_paths = self.paths[num_images:2 * num_images]

    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms(t_img, r_img,self.patchsize)
        syn_model = self.syn_model
        t_img, r_img, m_img = syn_model(t_img, r_img)

        T = to_tensor(t_img)
        R = to_tensor(r_img)
        M = to_tensor(m_img)

        return T, R, M, t_img, r_img, m_img

    def __getitem__(self, index):
        index_T = index % len(self.T_paths)
        index_R = index % len(self.R_paths)

        T_path = self.T_paths[index_T]
        R_path = self.R_paths[index_R]

        t_img = Image.open(T_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')

        T, R, M,t_img, r_img, m_img = self.data_synthesis(t_img, r_img)

        fn = os.path.basename(T_path)  # Extract the filename from the full path

       # Create a dictionary with non-None values only
        return {'input': M, 'target_t': T, 'target_r': R, 'fn': fn}

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.T_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.T_paths), len(self.R_paths))


class CEILTestDataset_R(BaseDataset):
    def __init__(self, datadir, datadir_R, fns=None, patchsize=(None, None), size=None,
                 enable_transforms=False):
        super(CEILTestDataset_R, self).__init__()
        self.size = size
        self.patchsize = patchsize
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))   # default: list all blended images
        self.datadir_R = join(self.datadir, datadir_R)
        self.enable_transforms = enable_transforms
        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]

        # Directly load images from disk (no preload)
        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        r_img = Image.open(join(self.datadir_R, fn)).convert('RGB')

        # Apply transforms if enabled
        if self.enable_transforms:
            t_img, m_img, r_img = triplet_data_transforms(t_img, m_img, r_img, patchsize=self.patchsize)

        # Convert to tensor
        T = to_tensor(t_img)
        M = to_tensor(m_img)
        R = to_tensor(r_img)

        # Build dictionary
        dic = {'input': M, 'target_t': T, 'target_r': R, 'fn': fn}

        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)



class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = fusion_ratios or [1./len(datasets)] * len(datasets)
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s'%(self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio/residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index%len(dataset)]
            residual -= ratio
    
    def __len__(self):
        return self.size

