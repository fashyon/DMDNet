import os
import settings
os.environ['CUDA_VISIBLE_DEVICES'] = settings.gpu_ids
from os.path import join
from model import Depth_Memory_Decoupling_Network as mynet
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import data.reflect_dataset as datasets
import util.util as util
import util.index as index
from util.visualizer import Visualizer

import time
import numpy as np
import random
from PIL import Image

from depth_estimation.MiDaS.midas.dpt_depth import DPTDepthModelWithFeatures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('os.environ["CUDA_VISIBLE_DEVICES"]:', os.environ["CUDA_VISIBLE_DEVICES"])

cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(settings.seed)
np.random.seed(settings.seed)
random.seed(settings.seed)

# -------------------------
# datasets
# -------------------------

eval_dataset_nature20 = datasets.CEILTestDataset_R(join(settings.datadir_test, 'Nature'), 'reflection_layer')
eval_dataset_real20 = datasets.CEILTestDataset_R(join(settings.datadir_test, 'real20_420'), 'reflection_layer')

eval_dataloader_nature20 = datasets.DataLoader(
    eval_dataset_nature20,
    batch_size=settings.batchsize,
    shuffle=False,
    num_workers=settings.nThreads,
    pin_memory=True
)
eval_dataloader_real20 = datasets.DataLoader(
    eval_dataset_real20,
    batch_size=settings.batchsize,
    shuffle=False,
    num_workers=settings.nThreads,
    pin_memory=True
)

eval_dataset_wild55 = datasets.CEILTestDataset_R(join(settings.datadir_test, 'SIR2/WildSceneDataset'), 'reflection_layer')
eval_dataloader_wild55 = datasets.DataLoader(
    eval_dataset_wild55,
    batch_size=settings.batchsize,
    shuffle=False,
    num_workers=settings.nThreads,
    pin_memory=True
)

eval_dataset_solid200 = datasets.CEILTestDataset_R(join(settings.datadir_test, 'SIR2/SolidObjectDataset'), 'reflection_layer')
eval_dataloader_solid200 = datasets.DataLoader(
    eval_dataset_solid200,
    batch_size=settings.batchsize,
    shuffle=False,
    num_workers=settings.nThreads,
    pin_memory=True
)

eval_dataset_postcard199 = datasets.CEILTestDataset_R(join(settings.datadir_test, 'SIR2/PostcardDataset'), 'reflection_layer')
eval_dataloader_postcard199 = datasets.DataLoader(
    eval_dataset_postcard199,
    batch_size=settings.batchsize,
    shuffle=False,
    num_workers=settings.nThreads,
    pin_memory=True
)


# -------------------------
# utils
# -------------------------
def tensor2im(image_tensor):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy


class myModel():
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_ids = settings.gpu_ids
        self.save_dir = os.path.join(settings.checkpoints_dir, settings.name)
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

        # ✅ test.py 默认不训练
        self.isTrain = False

    # -------- memory helpers (same as your modified train.py) --------
    def _tree_map(self, obj, fn):
        """递归地对 obj 里的 Tensor 应用 fn；支持 dict/list/tuple"""
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

    def _mem_to_cpu_detach(self, mem):
        def _fn(t: torch.Tensor):
            return t.detach().cpu()
        return self._tree_map(mem, _fn)

    def _move_memory_banks_to_device(self, device=None, dtype=None, non_blocking=True):
        device = device or self.device
        self.memory_banks = self._mem_to_device(
            self.memory_banks,
            device=device,
            dtype=dtype,
            non_blocking=non_blocking
        )

    def initialize(self):
        # 1) init memory banks (same specs as train.py)
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

        self.memory_banks = {}
        for name, dim in MEMORY_SPECS.items():
            mem = torch.rand((num_experts, dim, dim), dtype=torch.float)
            mem = F.normalize(mem, dim=1)
            self.memory_banks[name] = mem

        # 2) depth estimation
        if settings.model_type == "dpt_next_vit_large_384":
            self.depth_estimation = DPTDepthModelWithFeatures(
                path=settings.depth_estimation_model,
                backbone="next_vit_large_6m",
                non_negative=True,
                check_size=settings.check_size
            )
        self.depth_estimation.to(self.device)
        self.depth_estimation.eval()

        # 3) network
        if torch.cuda.device_count() == 1:
            print("one GPU running!")
            self.network = mynet().to(self.device)
        else:
            print("DataParallel GPU running!")
            self.network = torch.nn.DataParallel(mynet()).to(self.device)

        # 4) move memory to cuda
        self._move_memory_banks_to_device(self.device)

        # 5) load latest (or specified)
        self.load(self, settings.resume_epoch)

        # 6) print params
        self.print_param(self.network)

    def set_input(self, data, mode='eval'):
        mode = mode.lower()
        if mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
            target_t, target_r = None, None
        else:
            raise NotImplementedError

        if len(self.gpu_ids) > 0:
            input = input.to(device)
            if target_t is not None:
                target_t = target_t.to(device)
            if target_r is not None:
                target_r = target_r.to(device)

        self.input = input
        self.target_t = target_t
        self.target_r = target_r
        self.data_name = data_name

    def forward(self, train=False):
        depth_features_dict = self.depth_estimation(self.input)

        output_t, output_r, loss_mem_align_T, loss_mem_align_R, loss_mem_triplet_T, loss_mem_triplet_R, cof_T, cof_R, \
        self.memory_banks = self.network(self.input, depth_features_dict, self.memory_banks, train)

        self.output_t = output_t
        self.output_r = output_r
        return output_t

    def eval(self, data, savedir=None):
        self.network.eval()
        self.set_input(data, mode='eval')

        with torch.no_grad():
            self.forward(train=False)

            output_t = tensor2im(self.output_t)
            target_t = tensor2im(self.target_t)

            output_r = tensor2im(self.output_r) if self.target_r is not None else None
            target_r = tensor2im(self.target_r) if self.target_r is not None else None

            res = index.quality_assess(output_t, target_t)
            res_R = index.quality_assess(output_r, target_r) if self.target_r is not None else {}

            if savedir is not None:
                transmission_layer_dir = join(savedir, 'transmission_layer')
                reflection_layer_dir = join(savedir, 'reflection_layer')
                os.makedirs(transmission_layer_dir, exist_ok=True)
                os.makedirs(reflection_layer_dir, exist_ok=True)

                name, _ = os.path.splitext(os.path.basename(self.data_name[0]))
                Image.fromarray(output_t.astype(np.uint8)).save(
                    join(transmission_layer_dir, f"{name}_T_ssim{res['SSIM']:.6f}_psnr{res['PSNR']:.6f}.png")
                )
                if self.target_r is not None:
                    Image.fromarray(output_r.astype(np.uint8)).save(
                        join(reflection_layer_dir, f"{name}_R_ssim{res_R['SSIM']:.6f}_psnr{res_R['PSNR']:.6f}.png")
                    )

            return res, res_R

    @staticmethod
    def load(model, resume_epoch=None):
        # 1) load memory banks
        mem_path = os.path.join(model.save_dir, "memory_banks_latest.pt")
        if os.path.isfile(mem_path):
            mem = torch.load(mem_path, map_location="cpu")
            assert isinstance(mem, dict), f"memory file is not a dict: {mem_path}"
            model.memory_banks = model._mem_to_device(mem, device=model.device, dtype=None, non_blocking=True)
            print(f"[OK] loaded memory_banks: {mem_path} (num_keys={len(model.memory_banks)})")
        else:
            print(f"[WARN] memory_banks not found: {mem_path} (use initialized memory)")

        # 2) load model checkpoint
        model_path = util.get_model_list(model.save_dir, "model", epoch=resume_epoch)
        if model_path is None or (not os.path.isfile(model_path)):
            print(f"[WARN] model checkpoint not found (epoch={resume_epoch}), keep current weights.")
            return None

        state = torch.load(model_path, map_location="cpu")
        model.epoch = state.get("epoch", 0)
        model.iterations = state.get("iterations", 0)
        model.network.load_state_dict(state["network"], strict=True)

        print("Model successfully loaded:", model_path)
        print("Resume from epoch %d, iteration %d" % (model.epoch, model.iterations))
        return state

    def print_param(self, net):
        num_params = 0
        for p in net.parameters():
            num_params += p.numel()
        print('Total number of parameters: %d' % num_params)


class Engine(object):
    def __init__(self):
        self.PSNR_AvgdataTotal = 0
        self.SSIM_AvgdataTotal = 0
        self.Y_PSNR_AvgdataTotal = 0
        self.Y_SSIM_AvgdataTotal = 0
        self.LPIPS_AvgdataTotal = 0

        self.PSNR_AvgdataTotal_R = 0
        self.SSIM_AvgdataTotal_R = 0
        self.Y_PSNR_AvgdataTotal_R = 0
        self.Y_SSIM_AvgdataTotal_R = 0
        self.LPIPS_AvgdataTotal_R = 0

        self.Num_EvalDataset = 0
        self.__setup()

    def __setup(self):
        self.basedir = join(settings.checkpoints_dir, settings.name)
        print('self.basedir--------------------', self.basedir)
        os.makedirs(self.basedir, exist_ok=True)

        self.model = myModel()
        self.model.initialize()

        if not settings.no_log:
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))
            self.visualizer = Visualizer()

    def eval(self, val_loader, dataset_name, savedir=None, **kwargs):
        avg_meters = util.AverageMeters()
        model = self.model

        with torch.no_grad():
            PSNR_total = SSIM_total = 0.0
            PSNR_total_R = SSIM_total_R = 0.0
            Y_PSNR_total = Y_SSIM_total = LPIPS_total = 0.0
            Y_PSNR_total_R = Y_SSIM_total_R = LPIPS_total_R = 0.0

            for i, data in enumerate(val_loader):
                index_T, index_R = model.eval(data, savedir=savedir, **kwargs)
                avg_meters.update(index_T)

                PSNR_total += index_T['PSNR']
                SSIM_total += index_T['SSIM']
                Y_PSNR_total += index_T['Y_PSNR']
                Y_SSIM_total += index_T['Y_SSIM']
                LPIPS_total += index_T['LPIPS']

                if index_R:
                    PSNR_total_R += index_R['PSNR']
                    SSIM_total_R += index_R['SSIM']
                    Y_PSNR_total_R += index_R['Y_PSNR']
                    Y_SSIM_total_R += index_R['Y_SSIM']
                    LPIPS_total_R += index_R['LPIPS']

                util.progress_bar(i, len(val_loader), str(avg_meters))

        N = len(val_loader)

        average_PSNR_value = round(PSNR_total / N, 6)
        average_SSIM_value = round(SSIM_total / N, 6)
        average_Y_PSNR = round(Y_PSNR_total / N, 6)
        average_Y_SSIM = round(Y_SSIM_total / N, 6)
        average_LPIPS = round(LPIPS_total / N, 6)

        average_PSNR_value_R = round(PSNR_total_R / N, 6)
        average_SSIM_value_R = round(SSIM_total_R / N, 6)
        average_Y_PSNR_R = round(Y_PSNR_total_R / N, 6)
        average_Y_SSIM_R = round(Y_SSIM_total_R / N, 6)
        average_LPIPS_R = round(LPIPS_total_R / N, 6)

        print('average PSNR {}, average SSIM {}, on {} test_imgs ({})'.format(
            average_PSNR_value, average_SSIM_value, N, dataset_name))
        print('average Y_PSNR {}, Y_SSIM {}, LPIPS {}'.format(
            average_Y_PSNR, average_Y_SSIM, average_LPIPS))
        print('average PSNR_R {}, average SSIM_R {}, on {} test_imgs ({})'.format(
            average_PSNR_value_R, average_SSIM_value_R, N, dataset_name))
        print('average Y_PSNR_R {}, Y_SSIM_R {}, LPIPS_R {}'.format(
            average_Y_PSNR_R, average_Y_SSIM_R, average_LPIPS_R))

        self.PSNR_AvgdataTotal += average_PSNR_value
        self.SSIM_AvgdataTotal += average_SSIM_value
        self.Y_PSNR_AvgdataTotal += average_Y_PSNR
        self.Y_SSIM_AvgdataTotal += average_Y_SSIM
        self.LPIPS_AvgdataTotal += average_LPIPS

        self.PSNR_AvgdataTotal_R += average_PSNR_value_R
        self.SSIM_AvgdataTotal_R += average_SSIM_value_R
        self.Y_PSNR_AvgdataTotal_R += average_Y_PSNR_R
        self.Y_SSIM_AvgdataTotal_R += average_Y_SSIM_R
        self.LPIPS_AvgdataTotal_R += average_LPIPS_R

        logfile = open(self.basedir + '/loss_log.txt', 'a+')
        logfile.write(
            'step = {}, epoch = {}, '
            'PSNR = {}, SSIM = {}, Y_PSNR = {}, Y_SSIM = {}, LPIPS = {}, on {} test_imgs ({})\n'.format(
                self.iterations, self.epoch,
                average_PSNR_value, average_SSIM_value,
                average_Y_PSNR, average_Y_SSIM, average_LPIPS,
                len(val_loader), dataset_name
            )
        )
        logfile.write(
            'step = {}, epoch = {}, '
            'PSNR_R = {}, SSIM_R = {}, Y_PSNR_R = {}, Y_SSIM_R = {}, LPIPS_R = {}, on {} test_imgs ({})\n'.format(
                self.iterations, self.epoch,
                average_PSNR_value_R, average_SSIM_value_R,
                average_Y_PSNR_R, average_Y_SSIM_R, average_LPIPS_R,
                len(val_loader), dataset_name
            )
        )
        logfile.close()

        self.Num_EvalDataset += 1
        return avg_meters

    @property
    def iterations(self):
        return self.model.iterations

    @property
    def epoch(self):
        return self.model.epoch


engine = Engine()
result_dir = './results_eval'


if __name__ == '__main__':
    print("GPUs available:", torch.cuda.device_count())

    # reset totals
    engine.PSNR_AvgdataTotal = engine.SSIM_AvgdataTotal = 0
    engine.Y_PSNR_AvgdataTotal = engine.Y_SSIM_AvgdataTotal = engine.LPIPS_AvgdataTotal = 0
    engine.PSNR_AvgdataTotal_R = engine.SSIM_AvgdataTotal_R = 0
    engine.Y_PSNR_AvgdataTotal_R = engine.Y_SSIM_AvgdataTotal_R = engine.LPIPS_AvgdataTotal_R = 0
    engine.Num_EvalDataset = 0

    st = time.time()

    engine.eval(eval_dataloader_nature20, dataset_name='testdata_nature20', savedir=join(result_dir, 'Nature'))
    engine.eval(eval_dataloader_real20, dataset_name='testdata_real20', savedir=join(result_dir, 'real20_420'))
    engine.eval(eval_dataloader_wild55, dataset_name='testdata_wild55', savedir=join(result_dir, 'WildSceneDataset'))
    engine.eval(eval_dataloader_postcard199, dataset_name='testdata_postcard199', savedir=join(result_dir, 'PostcardDataset'))
    engine.eval(eval_dataloader_solid200, dataset_name='testdata_solid200', savedir=join(result_dir, 'SolidObjectDataset'))

    logfile = open(engine.basedir + '/loss_log.txt', 'a+')
    Num_EvalDataset = engine.Num_EvalDataset

    Avg_PSNR_AllData = engine.PSNR_AvgdataTotal / Num_EvalDataset
    Avg_SSIM_AllData = engine.SSIM_AvgdataTotal / Num_EvalDataset
    Avg_Y_PSNR_AllData = engine.Y_PSNR_AvgdataTotal / Num_EvalDataset
    Avg_Y_SSIM_AllData = engine.Y_SSIM_AvgdataTotal / Num_EvalDataset
    Avg_LPIPS_AllData = engine.LPIPS_AvgdataTotal / Num_EvalDataset

    Avg_PSNR_AllData_R = engine.PSNR_AvgdataTotal_R / Num_EvalDataset
    Avg_SSIM_AllData_R = engine.SSIM_AvgdataTotal_R / Num_EvalDataset
    Avg_Y_PSNR_AllData_R = engine.Y_PSNR_AvgdataTotal_R / Num_EvalDataset
    Avg_Y_SSIM_AllData_R = engine.Y_SSIM_AvgdataTotal_R / Num_EvalDataset
    Avg_LPIPS_AllData_R = engine.LPIPS_AvgdataTotal_R / Num_EvalDataset

    logfile.write(
        'step = {}, epoch = {}, Avg_PSNR_T = {}, Avg_SSIM_T = {}, Y_PSNR_T = {}, Y_SSIM_T = {}, LPIPS_T = {}, on {} Dataset\n'.format(
            engine.iterations, engine.epoch,
            round(Avg_PSNR_AllData, 6),
            round(Avg_SSIM_AllData, 6),
            round(Avg_Y_PSNR_AllData, 6),
            round(Avg_Y_SSIM_AllData, 6),
            round(Avg_LPIPS_AllData, 6),
            Num_EvalDataset
        )
    )
    logfile.write(
        'step = {}, epoch = {}, Avg_PSNR_R = {}, Avg_SSIM_R = {}, Y_PSNR_R = {}, Y_SSIM_R = {}, LPIPS_R = {}, on {} Dataset_R\n\n'.format(
            engine.iterations, engine.epoch,
            round(Avg_PSNR_AllData_R, 6),
            round(Avg_SSIM_AllData_R, 6),
            round(Avg_Y_PSNR_AllData_R, 6),
            round(Avg_Y_SSIM_AllData_R, 6),
            round(Avg_LPIPS_AllData_R, 6),
            Num_EvalDataset
        )
    )
    logfile.close()

    print('inference_time_on testdata cost: %.5f' % (time.time() - st))
