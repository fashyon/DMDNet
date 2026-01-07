import os
import settings
from model import Depth_Memory_Decoupling_Network as mynet
import losses as losses
os.environ['CUDA_VISIBLE_DEVICES'] = settings.gpu_ids
from os.path import join
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import util.index as index
from util.visualizer import Visualizer
import time
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import random
from PIL import Image
from collections import OrderedDict
from depth_estimation.MiDaS.midas.dpt_depth import DPTDepthModelWithFeatures
from torch.utils.tensorboard import SummaryWriter

print('os.environ["CUDA_VISIBLE_DEVICES"]:', os.environ["CUDA_VISIBLE_DEVICES"])

# Ensure reproducibility and optimize cudnn
cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(settings.seed)
np.random.seed(settings.seed)  # Random seed for numpy
random.seed(settings.seed)     # Random seed for Python random
# Initialize tensorboard writer before training
writer = SummaryWriter(log_dir=os.path.join(settings.checkpoints_dir, settings.name)+"/tensorboard")

# Construct datasets
train_dataset = datasets.CEILDataset(
    settings.datadir_VOC2012, util.read_fns(settings.datadir_VOC2012_txt),patchsize=settings.patchsize, size=settings.max_dataset_size, enable_transforms=True,shuffle=False,)
train_dataset_real = datasets.CEILTestDataset_R(settings.datadir_real,'reflection_layer',patchsize=settings.patchsize, enable_transforms=True)
train_dataset_nature = datasets.CEILTestDataset_R(settings.datadir_nature,'reflection_layer',patchsize=settings.patchsize, enable_transforms=True)

# Fusion dataset with weighted sampling from different sources
train_dataset_fusion = datasets.FusionDataset([train_dataset,train_dataset_real,train_dataset_nature], [0.6, 0.2, 0.2])

# DataLoader for training
train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=settings.batchsize, shuffle=not settings.serial_batches,
    num_workers=settings.nThreads, pin_memory=True)

# Validation datasets
eval_dataset_nature20 = datasets.CEILTestDataset_R(join(settings.datadir_test, 'Nature'),'reflection_layer')
eval_dataset_real20 = datasets.CEILTestDataset_R(join(settings.datadir_test, 'real20_420'),'reflection_layer')
eval_dataloader_nature20 = datasets.DataLoader(eval_dataset_nature20, batch_size=settings.batchsize, shuffle=False,
                                                    num_workers=settings.nThreads, pin_memory=True)
eval_dataloader_real20 = datasets.DataLoader(eval_dataset_real20, batch_size=settings.batchsize, shuffle=False,
                                                    num_workers=settings.nThreads, pin_memory=True)

# More datasets for evaluation
eval_dataset_wild55 = datasets.CEILTestDataset_R(join(settings.datadir_test, 'SIR2/WildSceneDataset'),'reflection_layer')
eval_dataloader_wild55 = datasets.DataLoader(eval_dataset_wild55, batch_size=settings.batchsize, shuffle=False,
                                               num_workers=settings.nThreads, pin_memory=True)
eval_dataset_solid200 = datasets.CEILTestDataset_R(join(settings.datadir_test, 'SIR2/SolidObjectDataset'),'reflection_layer')
eval_dataloader_solid200 = datasets.DataLoader(eval_dataset_solid200, batch_size=settings.batchsize, shuffle=False,
                                               num_workers=settings.nThreads, pin_memory=True)
eval_dataset_postcard199 = datasets.CEILTestDataset_R(join(settings.datadir_test, 'SIR2/PostcardDataset'),'reflection_layer')
eval_dataloader_postcard199 = datasets.DataLoader(eval_dataset_postcard199, batch_size=settings.batchsize, shuffle=False,
                                                  num_workers=settings.nThreads, pin_memory=True)


""" Utility Functions """
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
        self._count = 0
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.isTrain = True

    def initialize(self):
        self.vgg = None
        self.l1_loss = torch.nn.L1Loss()

        # =====================================================
        # 1. init memory banks
        # =====================================================
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

        # =====================================================
        # 2. depth estimation model
        # =====================================================
        if settings.model_type == "dpt_next_vit_large_384":
            self.depth_estimation = DPTDepthModelWithFeatures(
                path=settings.depth_estimation_model,
                backbone="next_vit_large_6m",
                non_negative=True,
                check_size=settings.check_size
            )

        self.depth_estimation.to(self.device)
        self.depth_estimation.eval()

        # =====================================================
        # 3. network & device
        # =====================================================
        self.vgg = losses.Vgg19(requires_grad=False).to(self.device)

        if torch.cuda.device_count() == 1:
            print("one GPU running!")
            self.network = mynet().to(self.device)
        elif torch.cuda.device_count() > 1:
            print("DataParallel GPU running!")
            self.network = torch.nn.DataParallel(mynet()).to(self.device)

        # ✅ 统一把 memory_banks 放到 cuda
        self._move_memory_banks_to_device(self.device)

        # =====================================================
        # 4. optimizer & loss
        # =====================================================
        if self.isTrain:
            self.vgg_loss = losses.VGGLoss(self.vgg)
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=settings.lr,
                betas=(0.9, 0.999),
                weight_decay=settings.wd
            )
            self._init_optimizer([self.optimizer])

        # =====================================================
        # 5. resume & print
        # =====================================================
        self.load(self, settings.resume_epoch)
        self.print_param(self.network)

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
        return obj  # 其它类型原样返回

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
        self.memory_banks = self._mem_to_device(self.memory_banks, device=device, dtype=dtype,
                                                non_blocking=non_blocking)

    def check_image_size(self, x):
        """Pad the input image so that it is divisible by check_size."""
        _, _, h, w = x.size()
        size = settings.check_size
        mod_pad_h = (size - h % size) % size
        mod_pad_w = (size - w % size) % size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        ori_size = [h, w]
        return x, ori_size

    def forward(self,train):
        """Forward pass with depth features and memory modules."""
        depth_features_dict  = self.depth_estimation(self.input)

        output_t, output_r, loss_mem_align_T, loss_mem_align_R, loss_mem_triplet_T, loss_mem_triplet_R, cof_T, cof_R, \
        self.memory_banks = self.network(self.input, depth_features_dict, self.memory_banks, train)

        # Save losses and coefficients
        self.loss_mem_align_T = loss_mem_align_T
        self.loss_mem_align_R = loss_mem_align_R
        self.loss_mem_triplet_T = loss_mem_triplet_T
        self.loss_mem_triplet_R = loss_mem_triplet_R
        self.cof_T_list = cof_T
        self.cof_R_list = cof_R

        # Save outputs
        self.output_t = output_t
        self.output_r = output_r
        return output_t


    def backward(self):
        """
        Compute the total loss and perform backpropagation.
        The loss is composed of:
          - Appearance loss (L1 + VGG perceptual loss)
          - Load loss (regularization on coefficient distribution)
          - Memory loss (alignment + triplet constraints)
        """
        self.loss_app = self.l1_loss(self.output_t, self.target_t) * settings.lambda_L1_T + self.l1_loss(self.output_r, self.target_r) * settings.lambda_L1_R\
                        + self.vgg_loss(self.output_t, self.target_t) * settings.lambda_vgg_T
        self.loss_load = losses.CVLoss(self.cof_T_list)* settings.lambda_load_T + losses.CVLoss(self.cof_R_list) * settings.lambda_load_R
        self.loss_mem = self.loss_mem_align_T * settings.lambda_align_T + self.loss_mem_triplet_T * settings.lambda_triplet_T + \
                        self.loss_mem_align_R * settings.lambda_align_R + self.loss_mem_triplet_R * settings.lambda_triplet_R
        self.loss_total = self.loss_app + self.loss_mem + self.loss_load

        # Backward pass
        self.loss_total.backward()

    def get_current_errors(self):
        """Return current loss values as an ordered dictionary."""
        ret_errors = OrderedDict()
        ret_errors['loss_total'] = self.loss_total.item()
        ret_errors['loss_app'] = self.loss_app.item()
        ret_errors['loss_load'] = self.loss_load.item()
        ret_errors['loss_mem'] = self.loss_mem.item()

        return ret_errors


    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        mode = mode.lower()
        if mode == 'train':
            input, target_t, target_r = data['input'], data['target_t'], data['target_r']
        elif mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)

        # Transfer data to GPU if available
        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device)  # (device=self.gpu_ids[0])
            if target_t is not None:
                target_t = target_t.to(device)  # to(device=self.gpu_ids[0])
            if target_r is not None:
                target_r = target_r.to(device)  # to(device=self.gpu_ids[0])

        self.input = input
        self.target_t = target_t
        self.target_r = target_r
        self.data_name = data_name

    def eval(self, data, savedir=None):
        """
        Evaluation mode: run forward pass, compute metrics, and optionally save results.
        Only the first image in a minibatch is processed.
        """
        self.network.eval()
        self.set_input(data, 'eval')
        with torch.no_grad():
            self.forward(train=False)
            output_t = tensor2im(self.output_t)
            target_t = tensor2im(self.target_t)
            if self.target_r is not None:
                output_r = tensor2im(self.output_r)
                target_r = tensor2im(self.target_r)

            # Compute quality metrics if aligned
            res = index.quality_assess(output_t, target_t)
            if self.target_r is not None:
                res_R = index.quality_assess(output_r, target_r)


            # Save results if requested
            if savedir is not None:
                transmission_layer_dir = join(savedir, 'transmission_layer')
                reflection_layer_dir = join(savedir, 'reflection_layer')
                vis_info_dir = join(savedir, 'vis_info')
                if not os.path.exists(transmission_layer_dir):
                    os.makedirs(transmission_layer_dir)
                if not os.path.exists(reflection_layer_dir):
                    os.makedirs(reflection_layer_dir)
                if not os.path.exists(vis_info_dir):
                    os.makedirs(vis_info_dir)
                name, file_extension = os.path.splitext(os.path.basename(self.data_name[0]))
                Image.fromarray(output_t.astype(np.uint8)).save(join(savedir,
                                                                     'transmission_layer/{}_T_ssim{:.6f}_psnr{:.6f}.png'.format(
                                                                         name, res['SSIM'], res['PSNR'])))
                Image.fromarray(output_r.astype(np.uint8)).save(join(savedir,
                                                                     'reflection_layer/{}_R_ssim{:.6f}_psnr{:.6f}.png'.format(
                                                                         name, res_R['SSIM'], res_R['PSNR'])))
            return res, res_R

    def save(self, label=None):
        epoch = self.epoch
        iterations = self.iterations

        if label is None:
            model_name = os.path.join(self.save_dir, f"model_{epoch:03d}_{iterations:08d}.pt")
            mem_name = os.path.join(self.save_dir, f"memory_banks_{epoch:03d}_{iterations:08d}.pt")
        else:
            model_name = os.path.join(self.save_dir, f"model_{label}.pt")
            mem_name = os.path.join(self.save_dir, f"memory_banks_{label}.pt")

        # 1) save model checkpoint
        ckpt = {
            "epoch": self.epoch,
            "iterations": self.iterations,
            "network": self.network.state_dict(),
            "opt": self.optimizer.state_dict() if self.isTrain else None,
        }
        torch.save(ckpt, model_name)

        # 2) save memory_banks
        mem_cpu = self._mem_to_cpu_detach(self.memory_banks)
        torch.save(mem_cpu, mem_name)

        # 3) 也可以同时存 latest（方便 load）
        torch.save(ckpt, os.path.join(self.save_dir, "model_latest.pt"))
        torch.save(mem_cpu, os.path.join(self.save_dir, "memory_banks_latest.pt"))

    def _init_optimizer(self, optimizers):
        """Initialize optimizer hyperparameters such as LR and weight decay."""
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', settings.lr)
            util.set_opt_param(optimizer, 'weight_decay', settings.wd)

    def optimize_parameters(self):
        """Perform one optimization step (forward + backward + update)."""
        self.network.train()
        self.forward(train=True)

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def print_param(self,net):
        """Print total number of trainable parameters in the network."""
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)

    @staticmethod
    def load(model, resume_epoch=None):
        # -------- 1) load memory banks (pure dict) --------
        mem_path = os.path.join(model.save_dir, "memory_banks_latest.pt")
        if os.path.isfile(mem_path):
            mem = torch.load(mem_path, map_location="cpu")  # mem is a dict
            assert isinstance(mem, dict), f"memory file is not a dict: {mem_path}"

            # move to device
            model.memory_banks = model._mem_to_device(mem, device=model.device, dtype=None, non_blocking=True)
            print(f"[OK] loaded memory_banks: {mem_path} (num_keys={len(model.memory_banks)})")
        else:
            print(f"[WARN] memory_banks not found: {mem_path} (use initialized memory)")

        # -------- 2) load model checkpoint --------
        model_path = util.get_model_list(model.save_dir, "model", epoch=resume_epoch)
        if model_path is None or (not os.path.isfile(model_path)):
            print(f"[WARN] model checkpoint not found (epoch={resume_epoch}), keep current weights.")
            return None

        state = torch.load(model_path, map_location="cpu")
        model.epoch = state.get("epoch", 0)
        model.iterations = state.get("iterations", 0)

        model.network.load_state_dict(state["network"], strict=True)

        if model.isTrain and state.get("opt", None) is not None:
            try:
                model.optimizer.load_state_dict(state["opt"])
            except Exception as e:
                print(f"[WARN] optimizer load failed: {e}")

        print("Model successfully loaded:", model_path)
        print("Resume from epoch %d, iteration %d" % (model.epoch, model.iterations))
        return state

class Engine(object):
    def __init__(self):
        # Average metrics for Transmission layer
        self.PSNR_AvgdataTotal = 0
        self.SSIM_AvgdataTotal = 0
        self.Y_PSNR_AvgdataTotal = 0
        self.Y_SSIM_AvgdataTotal = 0
        self.LPIPS_AvgdataTotal = 0

        # Average metrics for Reflection layer
        self.PSNR_AvgdataTotal_R = 0
        self.SSIM_AvgdataTotal_R = 0
        self.Y_PSNR_AvgdataTotal_R = 0
        self.Y_SSIM_AvgdataTotal_R = 0
        self.LPIPS_AvgdataTotal_R = 0

        # Number of evaluation datasets (used for averaging across datasets)
        self.Num_EvalDataset = 0
        self.__setup()

    def __setup(self):
        """Initialize model, loggers, and checkpoint directory."""
        self.basedir = join(settings.checkpoints_dir, settings.name)
        print('self.basedir--------------------', self.basedir)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        """Model"""
        self.model = myModel()
        self.model.initialize()
        if not settings.no_log:
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))
            self.visualizer = Visualizer()

    def train(self, train_loader, **kwargs):
        """Training loop for one epoch."""
        print('\nEpoch begin: %d' % (self.epoch + 1), 'len(train_loader): %d' % len(train_loader))
        avg_meters = util.AverageMeters()
        model = self.model
        epoch_start_time = time.time()

        if self.epoch ==0:
            model.save()
        for i, data in enumerate(train_loader):
            model.set_input(data, mode='train')
            model.optimize_parameters(**kwargs)
            errors = model.get_current_errors()
            # Record losses into tensorboard
            global_step = self.iterations  # 或者其它你设定的 iteration 计数
            for loss_name, loss_val in errors.items():
                writer.add_scalar(f"Loss/{loss_name}", loss_val, global_step)
            avg_meters.update(errors)
            print(f"step: {i + 1}/{len(train_loader)}; epoch: {self.epoch + 1}; lr: {self.model.optimizer.param_groups[0]['lr']:.7f}; loss: {self.model.loss_total.item():.6f}")
            self.iterations += 1
        self.epoch += 1

        # Save models and write logs
        if not settings.no_log:
            if self.epoch % settings.save_epoch_freq == 0:
                print('saving the model at epoch %d, iters %d' %
                      (self.epoch, self.iterations))
                model.save()
            print('saving the latest model at the end of epoch %d, iters %d' %
                  (self.epoch, self.iterations))
            model.save(label='latest')

            print('Time Taken: %d sec' %
                  (time.time() - epoch_start_time))

        # Reset dataset iterator if applicable
        train_loader.reset()

    def eval(self, val_loader, dataset_name, savedir=None, loss_key=None, **kwargs):
        """
        Evaluate model on validation set.
        Computes PSNR, SSIM, Y-channel PSNR/SSIM, and LPIPS for both Transmission and Reflection layers.
        """
        avg_meters = util.AverageMeters()
        model = self.model
        with torch.no_grad():
            # Accumulators for metrics
            PSNR_total = 0.0
            SSIM_total = 0.0
            PSNR_total_R = 0.0
            SSIM_total_R = 0.0
            # ✅ 新增指标累加器
            Y_PSNR_total = 0.0
            Y_SSIM_total = 0.0
            LPIPS_total = 0.0
            Y_PSNR_total_R = 0.0
            Y_SSIM_total_R = 0.0
            LPIPS_total_R = 0.0
            for i, data in enumerate(val_loader):
                index, index_R = model.eval(data, savedir=savedir, **kwargs)
                avg_meters.update(index)
                PSNR_total += index['PSNR']
                SSIM_total += index['SSIM']
                PSNR_total_R += index_R['PSNR']
                SSIM_total_R += index_R['SSIM']
                Y_PSNR_total += index['Y_PSNR']
                Y_SSIM_total += index['Y_SSIM']
                LPIPS_total += index['LPIPS']
                Y_PSNR_total_R += index_R['Y_PSNR']
                Y_SSIM_total_R += index_R['Y_SSIM']
                LPIPS_total_R += index_R['LPIPS']
                util.progress_bar(i, len(val_loader), str(avg_meters))
        N = len(val_loader)
        # Compute averages
        average_PSNR_value = round(PSNR_total / N, 6)
        average_SSIM_value = round(SSIM_total / N, 6)
        average_PSNR_value_R = round(PSNR_total_R / N, 6)
        average_SSIM_value_R = round(SSIM_total_R / N, 6)
        average_Y_PSNR = round(Y_PSNR_total / N, 6)
        average_Y_SSIM = round(Y_SSIM_total / N, 6)
        average_LPIPS = round(LPIPS_total / N, 6)
        average_Y_PSNR_R = round(Y_PSNR_total_R / N, 6)
        average_Y_SSIM_R = round(Y_SSIM_total_R / N, 6)
        average_LPIPS_R = round(LPIPS_total_R / N, 6)
        # Print results
        print('average PSNR {}, average SSIM {}, on {} test_imgs ({})'.format(
            average_PSNR_value, average_SSIM_value, N, dataset_name))
        print('average Y_PSNR {}, Y_SSIM {}, LPIPS {}'.format(
            average_Y_PSNR, average_Y_SSIM, average_LPIPS))
        print('average PSNR_R {}, average SSIM_R {}, on {} test_imgs ({})'.format(
            average_PSNR_value_R, average_SSIM_value_R, N, dataset_name))
        print('average Y_PSNR_R {}, Y_SSIM_R {}, LPIPS_R {}'.format(
            average_Y_PSNR_R, average_Y_SSIM_R, average_LPIPS_R))

        # Update cumulative values across datasets
        self.PSNR_AvgdataTotal += average_PSNR_value
        self.SSIM_AvgdataTotal += average_SSIM_value
        self.PSNR_AvgdataTotal_R += average_PSNR_value_R
        self.SSIM_AvgdataTotal_R += average_SSIM_value_R

        self.Y_PSNR_AvgdataTotal += average_Y_PSNR
        self.Y_SSIM_AvgdataTotal += average_Y_SSIM
        self.LPIPS_AvgdataTotal += average_LPIPS
        self.Y_PSNR_AvgdataTotal_R += average_Y_PSNR_R
        self.Y_SSIM_AvgdataTotal_R += average_Y_SSIM_R
        self.LPIPS_AvgdataTotal_R += average_LPIPS_R

        # Write logs to file
        logfile = open(self.basedir + '/loss_log.txt', 'a+')
        logfile.write('step = {}, epoch = {}, lr = {}, '
                      'PSNR = {}, SSIM = {}, Y_PSNR = {}, Y_SSIM = {}, LPIPS = {}, on {} test_imgs ({})\n'.format(
            self.iterations, self.epoch, model.optimizer.param_groups[0]['lr'],
            average_PSNR_value, average_SSIM_value,
            average_Y_PSNR, average_Y_SSIM, average_LPIPS,
            len(val_loader), dataset_name))

        logfile.write('step = {}, epoch = {}, lr = {}, '
                      'PSNR = {}, SSIM = {}, Y_PSNR = {}, Y_SSIM = {}, LPIPS = {}, on {} test_imgs ({})_R\n'.format(
            self.iterations, self.epoch, model.optimizer.param_groups[0]['lr'],
            average_PSNR_value_R, average_SSIM_value_R,
            average_Y_PSNR_R, average_Y_SSIM_R, average_LPIPS_R,
            len(val_loader), dataset_name))
        logfile.close()

        self.Num_EvalDataset += 1

        # Log to tensorboard if enabled
        if not settings.no_log:
            util.write_loss(self.writer, join('eval', dataset_name), avg_meters, self.epoch)

        # Save best model if selected metric improves
        if loss_key is not None:
            val_loss = avg_meters[loss_key]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print('saving the best model at the end of epoch %d, iters %d' % (self.epoch, self.iterations))
                model.save(label='best_{}_{}'.format(loss_key, dataset_name))

        return avg_meters

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.test(data, savedir=savedir, **kwargs)
                util.progress_bar(i, len(test_loader))

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e

engine = Engine()
def set_learning_rate(lr):
    """
    Update learning rate of the optimizer.
    """
    for optimizer in [engine.model.optimizer]:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

# define training strategy
if __name__ == '__main__':
    print("GPUs available:", torch.cuda.device_count())
    engine.eval(eval_dataloader_nature20, dataset_name='testdata_nature20')
    # Training loop
    while engine.epoch < settings.total_epochs:
        if engine.epoch == 30:
            set_learning_rate(5e-5)
        if engine.epoch == 50:
            set_learning_rate(1e-5)

        engine.train(train_dataloader_fusion)
        st = time.time()
        # Periodic evaluation and checkpoint saving
        if engine.epoch % settings.save_epoch_freq == 0:
            engine.PSNR_AvgdataTotal = 0
            engine.SSIM_AvgdataTotal = 0
            engine.PSNR_AvgdataTotal_R = 0
            engine.SSIM_AvgdataTotal_R = 0

            # Reset Y-channel and LPIPS metrics
            engine.Y_PSNR_AvgdataTotal = 0
            engine.Y_SSIM_AvgdataTotal = 0
            engine.LPIPS_AvgdataTotal = 0
            engine.Y_PSNR_AvgdataTotal_R = 0
            engine.Y_SSIM_AvgdataTotal_R = 0
            engine.LPIPS_AvgdataTotal_R = 0

            engine.Num_EvalDataset = 0

            # Evaluate on all datasets
            engine.eval(eval_dataloader_nature20, dataset_name='testdata_nature20')
            engine.eval(eval_dataloader_real20, dataset_name='testdata_real20')
            engine.eval(eval_dataloader_wild55, dataset_name='testdata_wild55')
            engine.eval(eval_dataloader_postcard199, dataset_name='testdata_postcard199')
            engine.eval(eval_dataloader_solid200, dataset_name='testdata_solid200')

            logfile = open(engine.basedir + '/loss_log.txt', 'a+')
            Num_EvalDataset = engine.Num_EvalDataset

            # Compute averaged metrics across all evaluation datasets (Transmission layer)
            Avg_PSNR_AllData = engine.PSNR_AvgdataTotal / Num_EvalDataset
            Avg_SSIM_AllData = engine.SSIM_AvgdataTotal / Num_EvalDataset
            Avg_Y_PSNR_AllData = engine.Y_PSNR_AvgdataTotal / Num_EvalDataset
            Avg_Y_SSIM_AllData = engine.Y_SSIM_AvgdataTotal / Num_EvalDataset
            Avg_LPIPS_AllData = engine.LPIPS_AvgdataTotal / Num_EvalDataset

            # Compute averaged metrics across all evaluation datasets (Reflection layer)
            Avg_PSNR_AllData_R = engine.PSNR_AvgdataTotal_R / Num_EvalDataset
            Avg_SSIM_AllData_R = engine.SSIM_AvgdataTotal_R / Num_EvalDataset
            Avg_Y_PSNR_AllData_R = engine.Y_PSNR_AvgdataTotal_R / Num_EvalDataset
            Avg_Y_SSIM_AllData_R = engine.Y_SSIM_AvgdataTotal_R / Num_EvalDataset
            Avg_LPIPS_AllData_R = engine.LPIPS_AvgdataTotal_R / Num_EvalDataset

            # Write Transmission layer logs
            logfile.write(
                'step  = {}, epoch = {}, Avg_PSNR_T = {}, Avg_SSIM_T = {}, Y_PSNR_T = {}, Y_SSIM_T = {}, LPIPS_T = {}, on {} Dataset\n'.format(
                    engine.iterations, engine.epoch,
                    round(Avg_PSNR_AllData, 6),
                    round(Avg_SSIM_AllData, 6),
                    round(Avg_Y_PSNR_AllData, 6),
                    round(Avg_Y_SSIM_AllData, 6),
                    round(Avg_LPIPS_AllData, 6),
                    Num_EvalDataset
                ))

            # Write Reflection layer logs
            logfile.write(
                'step  = {}, epoch = {}, Avg_PSNR_R = {}, Avg_SSIM_R = {}, Y_PSNR_R = {}, Y_SSIM_R = {}, LPIPS_R = {}, on {} Dataset_R\n\n'.format(
                    engine.iterations, engine.epoch,
                    round(Avg_PSNR_AllData_R, 6),
                    round(Avg_SSIM_AllData_R, 6),
                    round(Avg_Y_PSNR_AllData_R, 6),
                    round(Avg_Y_SSIM_AllData_R, 6),
                    round(Avg_LPIPS_AllData_R, 6),
                    Num_EvalDataset
                ))

            logfile.close()
            print('inference_time_on testdata cost: %.5f' % (time.time() - st))


