from __future__ import division
import math
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import cv2
import torchvision.transforms as transforms
from scipy.signal import convolve2d

# utility
def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

to_tensor = transforms.ToTensor()

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


class ReflectionSythesis_1(object):
    """Reflection image data synthesis for weakly-supervised learning
    """
    def __init__(self, kernel_sizes=None, low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3):
        self.kernel_sizes = kernel_sizes or [11]
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.low_gamma = low_gamma
        self.high_gamma = high_gamma
        print('[i] reflection sythesis model: {}'.format({
            'kernel_sizes': kernel_sizes, 'low_sigma': low_sigma, 'high_sigma': high_sigma,
            'low_gamma': low_gamma, 'high_gamma': high_gamma}))

    def __call__(self, T, R):
        if not _is_pil_image(T):
            raise TypeError('T should be PIL Image. Got {}'.format(type(B)))
        if not _is_pil_image(R):
            raise TypeError('R should be PIL Image. Got {}'.format(type(R)))
        
        T_ = np.asarray(T, np.float32) / 255.
        R_ = np.asarray(R, np.float32) / 255.

        sigma = np.random.uniform(self.low_sigma, self.high_sigma)
        gamma = np.random.uniform(self.low_gamma, self.high_gamma)
        R_blur = R_
        kernel = cv2.getGaussianKernel(11, sigma)
        kernel2d = np.dot(kernel, kernel.T)

        for i in range(3):
            R_blur[...,i] = convolve2d(R_blur[...,i], kernel2d, mode='same')

        M_ = T_ + R_blur
        
        if np.max(M_) > 1:
            m = M_[M_ > 1]
            m = (np.mean(m) - 1) * gamma
            R_blur = np.clip(R_blur - m, 0, 1)
            M_ = np.clip(R_blur + T_, 0, 1)
        
        return T_, R_blur, M_


