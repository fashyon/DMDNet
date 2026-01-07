import os
import glob
import argparse
from os.path import join
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import settings
from model import Depth_Memory_Decoupling_Network as mynet
import util.util as util
from depth_estimation.MiDaS.midas.dpt_depth import DPTDepthModelWithFeatures


# -------------------------
# Utils
# -------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def is_image_file(p: str) -> bool:
    return p.lower().endswith(IMG_EXTS)


def list_images(path: str):
    if os.path.isdir(path):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(join(path, f"**/*{ext}"), recursive=True))
        files = sorted(set(files))
        return files
    else:
        return [path]


def tensor2im_01(t: torch.Tensor) -> np.ndarray:
    """
    t: [1, C, H, W], range [0,1] (assumed)
    return: HWC uint8
    """
    t = t.detach().float().clamp(0, 1)
    x = t[0].cpu().numpy()
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    x = (np.transpose(x, (1, 2, 0)) * 255.0).round().astype(np.uint8)
    return x


def load_image_as_tensor(path: str) -> torch.Tensor:
    """
    返回 [1,3,H,W] float32 in [0,1]
    注意：这里用最稳的 ToTensor 逻辑，避免额外依赖 torchvision。
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC [0,1]
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    t = torch.from_numpy(arr).unsqueeze(0)  # 1CHW
    return t


# -------------------------
# Model wrapper (match your train.py style)
# -------------------------
class ShowModel:
    def __init__(self, device):
        self.device = device
        self.gpu_ids = settings.gpu_ids
        self.save_dir = os.path.join(settings.checkpoints_dir, settings.name)

        self.network = None
        self.depth_estimation = None
        self.memory_banks = None

    # ---- tree utils (same idea as your train.py) ----
    def _tree_map(self, obj, fn):
        if torch.is_tensor(obj):
            return fn(obj)
        if isinstance(obj, dict):
            return {k: self._tree_map(v, fn) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._tree_map(v, fn) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._tree_map(v, fn) for v in obj)
        return obj

    def _mem_to_device(self, mem, device, dtype=None, non_blocking=True):
        def _fn(t: torch.Tensor):
            if dtype is not None:
                t = t.to(dtype=dtype)
            return t.to(device=device, non_blocking=non_blocking)
        return self._tree_map(mem, _fn)

    def _init_memory_banks_like_train(self):
        dims = settings.dims
        num_experts = settings.num_experts

        MEMORY_SPECS = {
            # memory（level1/2/3/4/5）
            "memory_level1_T": dims[0],
            "memory_level1_R": dims[0],
            "memory_level2_T": dims[1],
            "memory_level2_R": dims[1],
            "memory_level3_T": dims[2],
            "memory_level3_R": dims[2],
            "memory_level4_T": dims[3],
            "memory_level4_R": dims[3],
            "memory_level5_T": dims[4],
            "memory_level5_R": dims[4],

            # DS branch memories
            "memory_DS3_T1": dims[2],
            "memory_DS3_T2": dims[2],
            "memory_DS3_R1": dims[2],
            "memory_DS3_R2": dims[2],
            "memory_DS4_T1": dims[3],
            "memory_DS4_T2": dims[3],
            "memory_DS4_R1": dims[3],
            "memory_DS4_R2": dims[3],
            "memory_DS5_T1": dims[4],
            "memory_DS5_T2": dims[4],
            "memory_DS5_R1": dims[4],
            "memory_DS5_R2": dims[4],
        }
        mem = {}
        for name, dim in MEMORY_SPECS.items():
            m = torch.rand((num_experts, dim, dim), dtype=torch.float)
            m = F.normalize(m, dim=1)
            mem[name] = m
        return mem

    def _load_latest_ckpt_and_memory(self, ckpt_path=None, mem_path=None):
        if ckpt_path is None:
            ckpt_path = os.path.join(self.save_dir, "model_latest.pt")
        if mem_path is None:
            mem_path = os.path.join(self.save_dir, "memory_banks_latest.pt")

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[show.py] model ckpt not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location="cpu")
        self.network.load_state_dict(state["network"], strict=True)
        print(f"[OK] loaded model: {ckpt_path}")

        if os.path.isfile(mem_path):
            mem = torch.load(mem_path, map_location="cpu")
            if not isinstance(mem, dict):
                raise TypeError(f"[show.py] memory file must be dict, got {type(mem)}")
            self.memory_banks = self._mem_to_device(mem, device=self.device, dtype=None, non_blocking=True)
            print(f"[OK] loaded memory_banks: {mem_path} (keys={len(self.memory_banks)})")
        else:
            print(f"[WARN] memory_banks not found: {mem_path} -> use random init")
            self.memory_banks = self._init_memory_banks_like_train()
            self.memory_banks = self._mem_to_device(self.memory_banks, device=self.device, dtype=None, non_blocking=True)

    def initialize(self, ckpt_path=None, mem_path=None):
        # depth
        if settings.model_type == "dpt_next_vit_large_384":
            self.depth_estimation = DPTDepthModelWithFeatures(
                path=settings.depth_estimation_model,
                backbone="next_vit_large_6m",
                non_negative=True,
                check_size=settings.check_size
            ).to(self.device).eval()
        else:
            raise ValueError(f"[show.py] unsupported model_type: {settings.model_type}")

        # network
        if torch.cuda.device_count() <= 1:
            self.network = mynet().to(self.device)
        else:
            self.network = torch.nn.DataParallel(mynet()).to(self.device)

        self.network.eval()
        self._load_latest_ckpt_and_memory(ckpt_path=ckpt_path, mem_path=mem_path)

    def forward(self, x: torch.Tensor):
        """
        x: [1,3,H,W] in [0,1]
        return: output_t, output_r (maybe None)
        """
        x = x.to(self.device, non_blocking=True)
        with torch.no_grad():
            depth_features = self.depth_estimation(x)
            # 你的网络 forward: (input, depth_features_dict, memory_banks, train_flag)
            out = self.network(x, depth_features, self.memory_banks, False)

            # 兼容你 train.py 的返回格式：
            # output_t, output_r, ..., cof_T, cof_R, self.memory_banks
            output_t = out[0]
            output_r = out[1]
            self.memory_banks = out[-1]
        return output_t, output_r


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="single image path or a folder")
    parser.add_argument("--outdir", type=str, default="./results_show", help="output dir")
    parser.add_argument("--ckpt", type=str, default=None, help="override model_latest.pt path")
    parser.add_argument("--mem", type=str, default=None, help="override memory_banks_latest.pt path")
    parser.add_argument("--keep_structure", action="store_true",
                        help="if input is a folder, keep relative folder structure in outdir")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = settings.gpu_ids
    print('os.environ["CUDA_VISIBLE_DEVICES"]:', os.environ["CUDA_VISIBLE_DEVICES"])

    # cudnn & seed (same vibe as train.py)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShowModel(device=device)
    model.initialize(ckpt_path=args.ckpt, mem_path=args.mem)

    files = list_images(args.input)
    if len(files) == 0:
        raise FileNotFoundError(f"[show.py] no images found in: {args.input}")

    os.makedirs(args.outdir, exist_ok=True)
    base_root = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)

    for idx, p in enumerate(files):
        if not is_image_file(p):
            continue

        # output path
        if args.keep_structure and os.path.isdir(args.input):
            rel = os.path.relpath(p, base_root)
            rel_dir = os.path.dirname(rel)
            out_dir = join(args.outdir, rel_dir)
        else:
            out_dir = args.outdir
        os.makedirs(out_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(p))[0]
        out_T = join(out_dir, f"{stem}_T.png")
        out_R = join(out_dir, f"{stem}_R.png")

        x = load_image_as_tensor(p)  # [1,3,H,W] in [0,1]

        out_t, out_r = model.forward(x)

        Image.fromarray(tensor2im_01(out_t)).save(out_T)
        if out_r is not None:
            Image.fromarray(tensor2im_01(out_r)).save(out_R)

        print(f"[{idx+1}/{len(files)}] saved: {out_T}" + (f", {out_R}"))

    print(f"[DONE] results saved to: {args.outdir}")


if __name__ == "__main__":
    main()
