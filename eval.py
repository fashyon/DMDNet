"""
Evaluation script for PSNR / SSIM / LPIPS on PNG images.

Evaluation protocol (FIXED):
- All images are evaluated after being saved as PNG.
- Metrics:
    * PSNR (RGB)
    * SSIM (RGB)
    * PSNR (YCbCr-Y)
    * SSIM (YCbCr-Y)
    * LPIPS (AlexNet)
- LPIPS is computed on RGB images normalized to [-1, 1].

This script reproduces the evaluation results reported in the AAAI paper.
"""

import os
import cv2
import numpy as np
from natsort import natsorted

import torch
import lpips

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


# Root directories
RESULT_ROOT = r"../results"     # results
GT_ROOT = r"../GT"        # GT

# Datasets to be evaluated
DATASETS = [
    "Nature",
    "real20_420",
    "WildSceneDataset",
    "PostcardDataset",
    "SolidObjectDataset",
    "NightIRS",
]

# Layer types to evaluate
LAYERS = [
    "transmission_layer",
    "reflection_layer",
]

# LPIPS configuration
LPIPS_BACKBONE = "alex"

# =====================================================


# -----------------------------
# Metric functions
# -----------------------------
def calc_psnr_ssim(gt, pred):
    """Compute PSNR / SSIM in RGB and Y channel."""
    gt_y = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    pred_y = cv2.cvtColor(pred, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    psnr_y = compare_psnr(gt_y, pred_y)
    ssim_y = compare_ssim(gt_y, pred_y)

    psnr_rgb = compare_psnr(gt, pred)
    ssim_rgb = compare_ssim(gt, pred, channel_axis=-1)

    return psnr_rgb, ssim_rgb, psnr_y, ssim_y


def calc_lpips(gt, pred, lpips_fn, device):
    """LPIPS on RGB images normalized to [-1, 1]."""
    gt_rgb = gt[..., ::-1] / 127.5 - 1.0
    pred_rgb = pred[..., ::-1] / 127.5 - 1.0

    gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    pred_tensor = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        score = lpips_fn(gt_tensor, pred_tensor).item()
    return score


# -----------------------------
# Dataset evaluation
# -----------------------------
def evaluate_dataset(pred_dir, gt_dir, lpips_fn, device):
    pred_list = natsorted(os.listdir(pred_dir))
    gt_list = natsorted(os.listdir(gt_dir))

    assert len(pred_list) == len(gt_list), \
        f"Image number mismatch: {pred_dir} vs {gt_dir}"

    psnr_rgb_all, ssim_rgb_all = [], []
    psnr_y_all, ssim_y_all = [], []
    lpips_all = []

    for p_name, g_name in zip(pred_list, gt_list):
        pred = cv2.imread(os.path.join(pred_dir, p_name))
        gt = cv2.imread(os.path.join(gt_dir, g_name))

        if pred.shape != gt.shape:
            raise ValueError(f"Image size mismatch: {p_name} vs {g_name}")

        psnr_rgb, ssim_rgb, psnr_y, ssim_y = calc_psnr_ssim(gt, pred)
        lpips_score = calc_lpips(gt, pred, lpips_fn, device)

        psnr_rgb_all.append(psnr_rgb)
        ssim_rgb_all.append(ssim_rgb)
        psnr_y_all.append(psnr_y)
        ssim_y_all.append(ssim_y)
        lpips_all.append(lpips_score)

    return {
        "num": len(pred_list),
        "psnr_rgb": np.mean(psnr_rgb_all),
        "ssim_rgb": np.mean(ssim_rgb_all),
        "psnr_y": np.mean(psnr_y_all),
        "ssim_y": np.mean(ssim_y_all),
        "lpips": np.mean(lpips_all),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net=LPIPS_BACKBONE).to(device).eval()

    print("==========================================")
    print(" Evaluation on saved PNG images")
    print("==========================================")
    print(f"Device: {device}")
    print(f"LPIPS backbone: {LPIPS_BACKBONE}\n")

    for dataset in DATASETS:
        for layer in LAYERS:
            pred_dir = os.path.join(RESULT_ROOT, dataset, layer)
            gt_dir = os.path.join(GT_ROOT, dataset, layer)

            if not os.path.isdir(pred_dir):
                continue

            stats = evaluate_dataset(pred_dir, gt_dir, lpips_fn, device)

            print(
                f"[{dataset} | {layer}] "
                f"PSNR_RGB {stats['psnr_rgb']:.4f}, "
                f"SSIM_RGB {stats['ssim_rgb']:.4f}, "
                f"PSNR_Y {stats['psnr_y']:.4f}, "
                f"SSIM_Y {stats['ssim_y']:.4f}, "
                f"LPIPS {stats['lpips']:.4f} "
                f"({stats['num']} images)"
            )


if __name__ == "__main__":
    main()
