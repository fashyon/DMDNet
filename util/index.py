# Metrics/Indexes
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.color import rgb2ycbcr
from functools import partial
import numpy as np
import torch
import lpips
import math

# ==== Initialize LPIPS model ====
# LPIPS (Learned Perceptual Image Patch Similarity) is a perceptual metric
lpips_model = lpips.LPIPS(net='alex').cuda()  # Load LPIPS with AlexNet backbone ('vgg' is also available)
lpips_model.eval()  # Set the model to evaluation mode (no gradient updates)

class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = X[..., ch]
            y = Y[..., ch]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

def calcuate_PSNR(img1,img2):
    mse = np.mean((img1 - img2)**2)
    if mse ==0:
        return float('inf')
    return 20*math.log10(255.0/math.sqrt(mse))

cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=255))
cal_bwssim = Bandwise(partial(compare_ssim, data_range=255))


def compare_ncc(x, y):
    return np.mean((x-np.mean(x)) * (y-np.mean(y))) / (np.std(x) * np.std(y)) 


def ssq_error(correct, estimate):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate**2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate**2)
    else:
        alpha = 0.
    return np.sum((correct - alpha*estimate) ** 2)


def local_error(correct, estimate, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N, C = correct.shape
    ssq = total = 0.
    for c in range(C):
        for i in range(0, M - window_size + 1, window_shift):
            for j in range(0, N - window_size + 1, window_shift):
                correct_curr = correct[i:i+window_size, j:j+window_size, c]
                estimate_curr = estimate[i:i+window_size, j:j+window_size, c]
                ssq += ssq_error(correct_curr, estimate_curr)
                total += np.sum(correct_curr**2)
    # assert np.isnan(ssq/total)
    return ssq / total

def to_tensor(img):
    # 输入形状：[H,W,C]，输出：[1,C,H,W]，并归一化到[-1,1]
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return (img - 0.5) * 2

def quality_assess(X, Y):
    # Y: ground truth; X: estimate
    psnr = np.mean(cal_bwpsnr(Y, X))
    ssim = np.mean(cal_bwssim(Y, X))
    # lmse = local_error(Y, X, 20, 10)
    # ncc = compare_ncc(Y, X)

    # 转为 uint8 做 YCbCr
    X_uint8 = X.astype(np.uint8)
    Y_uint8 = Y.astype(np.uint8)
    Ycbcr_X = rgb2ycbcr(X_uint8)
    Ycbcr_Y = rgb2ycbcr(Y_uint8)
    X_Y = Ycbcr_X[..., 0]
    Y_Y = Ycbcr_Y[..., 0]

    y_psnr = compare_psnr(Y_Y, X_Y, data_range=255)
    y_ssim = compare_ssim(Y_Y, X_Y, data_range=255)

    # LPIPS
    X_tensor = to_tensor(X).cuda()
    Y_tensor = to_tensor(Y).cuda()
    with torch.no_grad():
        lpips_score = lpips_model(X_tensor, Y_tensor).item()

    return {
        'PSNR': psnr,
        'SSIM': ssim,
        # 'LMSE': lmse,
        # 'NCC': ncc,
        'Y_PSNR': y_psnr,
        'Y_SSIM': y_ssim,
        'LPIPS': lpips_score
    }