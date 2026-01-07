# Define network components here
import os
import settings
os.environ['CUDA_VISIBLE_DEVICES'] = settings.gpu_ids
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import cv2


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CABlock(nn.Module):
    def __init__(self, channels):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return x * self.ca(x)


class DualStreamGate(nn.Module):
    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2

class DualStreamSeq(nn.Sequential):
    def forward(self, x, y=None):
        y = y if y is not None else x
        for module in self:
            x, y = module(x, y)
        return x, y


class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super(DualStreamBlock, self).__init__()
        self.seq = nn.Sequential()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.seq.add_module(str(idx), module)

    def forward(self, x, y):
        return self.seq(x), self.seq(y)


class MuGIBlock(nn.Module):
    def __init__(self, c, shared_b=True):
        """
            MuGIBlock: Mutually-Gated Interactive Block
            Args:
                c (int): number of input/output channels
                shared_b (bool): whether to share parameter 'b' between two streams
            """
        super().__init__()

        # ==== First dual-stream sequence (block1) ====
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        # Learnable scaling parameters for skip connections in block1
        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # ==== Second dual-stream sequence (block2) ====
        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        # Learnable scaling parameters for final outputs
        self.shared_b = shared_b
        if shared_b:
            self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        else:
            self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_l, inp_r):
        x, y = self.block1(inp_l, inp_r)
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        if self.shared_b:
            out_l, out_r = x_skip + x * self.b, y_skip + y * self.b
        else:
            out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r


class Encoding_Branch(nn.Module):
    def __init__(self, dims, enc_blk_nums):
        """
        Encoding Branch of the network (dual-stream encoder).
        Args:
            dims (list[int]): channel dimensions for each stage
            enc_blk_nums (list[int]): number of MuGI blocks in each stage
        """
        super(Encoding_Branch, self).__init__()
        self.dims = dims
        # Basic upsampling and downsampling operators
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        self.down = nn.AvgPool2d(2,2)

        # Initial stem block: project 3-channel RGB input to dims[0]
        self.stem = DualStreamBlock(nn.Conv2d(3, dims[0], 3, padding=1))
        self.block1 = DualStreamSeq(
            *[MuGIBlock(dims[0]) for _ in range(enc_blk_nums[0])],
            DualStreamBlock(self.down),
            DualStreamBlock(nn.Conv2d(dims[0], dims[1], 1, 1))
        )
        self.block2 = DualStreamSeq(
            *[MuGIBlock(dims[1]) for _ in range(enc_blk_nums[1])],
            DualStreamBlock(self.down),
            DualStreamBlock(nn.Conv2d(dims[1], dims[2], 1, 1))
        )
        self.block3 = DualStreamSeq(
            *[MuGIBlock(dims[2]) for _ in range(enc_blk_nums[2])],
            DualStreamBlock(self.down),
            DualStreamBlock(nn.Conv2d(dims[2], dims[3], 1, 1))
        )
        self.block4 = DualStreamSeq(
            *[MuGIBlock(dims[3]) for _ in range(enc_blk_nums[3])],
            DualStreamBlock(self.down),
            DualStreamBlock(nn.Conv2d(dims[3], dims[4], 1, 1))
        )


    def forward(self, x):
        # (1, 48, 384, 384) (1, 96, 192, 192) (1, 192, 96, 96) (1, 384, 48, 48) (1, 384, 24, 24)
        x1, y1 = self.stem(x, x)
        x2, y2 = self.block1(x1, y1)
        x3, y3 = self.block2(x2, y2)
        x4, y4 = self.block3(x3, y3)
        x5, y5 = self.block4(x4, y4)
        return (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Depth_Synergized_Decoupling_Mamba(nn.Module):
    """
    Depth-Synergized Decoupling Mamba (DS-Mamba).
    This module enhances Mamba-based state-space modeling with depth-aware scanning and
    selective state updates to better separate transmission and reflection features.

    Key ideas:
    - Transmission (T) and Reflection (R) features are scanned with different depth-aware orders:
        * Transmission: Regional scanning (large area first, then near to far).
        * Reflection: Global scanning (all pixels sorted by depth globally).
    - Features are scanned in both forward and backward directions.
    - Depth features are fused with state-space dynamics (B, C matrices) to adaptively
      control feature propagation.
    - Positional encoding (sin-cos) is aligned with scanning order.
    """
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            ssm_ratio=2,
            dt_rank="auto",
            # ======================
            dropout=0.,
            conv_bias=True,
            bias=False,
            dtype=None,
            # ======================
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": dtype}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 2 # number of scanning directions (forward + backward)

        # Depth fusion with feature map
        self.fuse = nn.Sequential(nn.Conv2d(2*self.d_inner,1,3,1,1),nn.LeakyReLU(0.2))

        # Input projection: expand channels for state-space dynamics
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Depthwise Conv for local spatial modeling
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # Projection of input features to state-space params
        self.x_proj_channel = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight_channel = nn.Parameter(torch.stack([t.weight for t in self.x_proj_channel], dim=0))
        del self.x_proj_channel

        # Projection of Δt (time step) for selective scan
        self.dt_projs_channel = [
            self.dt_init(self.dt_rank,self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight_channel = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs_channel], dim=0))  # (K, 2*d_inner, dt_rank)
        self.dt_projs_bias_channel = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs_channel], dim=0))  # (K * 2*d_inner)
        del self.dt_projs_channel

        # State-space parameters A, D (initialized as learnable)
        self.A_logs_ch_T = self.A_log_init(self.d_state,  self.d_inner, copies=self.K, merge=True)
        self.A_logs_ch_R = self.A_log_init(self.d_state,  self.d_inner, copies=self.K, merge=True)
        self.Ds_ch_T = self.D_init(self.d_inner, copies=self.K, merge=True)
        self.Ds_ch_R = self.D_init(self.d_inner, copies=self.K, merge=True)

        # Output normalization and projection
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Core selective scan function (implements state-space update along scan order)
        self.selective_scan = selective_scan_fn

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=1),
            nn.LeakyReLU(0.2)
        )
        self.d_pos = self.d_inner

        # Depth projection for state updates
        self.depth_proj_weight = [
            nn.Linear(self.d_inner, (self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.depth_proj_weight_channel = nn.Parameter(torch.stack([t.weight for t in self.depth_proj_weight], dim=0))
        del self.depth_proj_weight

        self.conv_depth_sig = nn.Sequential(nn.Conv2d(self.d_inner, self.d_state,1,1),nn.LeakyReLU(0.2))

    def cross_scan_2d(self,x, order):
        if x.dim() == 4:
            B, C, H, W = x.shape
            L = H * W
            x_flat = x.view(B, C, L)  # Flatten spatial dimensions
        elif x.dim() == 3:
            B, C, L = x.shape
            x_flat = x
        else:
            raise ValueError("Input x must be 3D or 4D tensor")

        # Reorder sequence using order indices (expand order to match channel dimension)
        x_sorted = torch.gather(x_flat, dim=2, index=order.unsqueeze(1).expand(B, C, order.shape[1]))

        # Stack forward and reverse scan along a new axis
        x_scanned = torch.stack([x_sorted, x_sorted.flip(dims=[2])], dim=1)
        return x_scanned  # [B, 2, C, L']

    def Spatial_Positional_Encoding(self, H: int, W: int, d_pos: int, device):
        """
           Generate 2D sine-cosine positional embeddings for spatial positions.

           Args:
               H (int): Height of the feature map.
               W (int): Width of the feature map.
               d_pos (int): Desired embedding dimension.
               device (torch.device): Device to place the embeddings.

           Returns:
               emb (Tensor): Positional encoding of shape (H*W, d_pos).
                             Each location (h, w) is encoded by sine/cosine functions
                             of its normalized coordinates.
           """
        # Create normalized coordinate grid in range [0, 1]
        ys = torch.arange(H, device=device).float() / (H - 1)
        xs = torch.arange(W, device=device).float() / (W - 1)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]

        # Stack into (H, W, 2), representing (y, x)
        grid = torch.stack([grid_y, grid_x], dim=-1)

        # Compute inverse frequency for sine-cosine encoding
        dim_each = d_pos // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(dim_each, device=device).float() / dim_each))

        # Apply sinusoidal encoding for x and y axes separately
        pos_x = grid[..., 1:2] * inv_freq.view(1, 1, -1)
        pos_y = grid[..., 0:1] * inv_freq.view(1, 1, -1)

        emb_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)  # [H,W,2*dim_each]
        emb_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1)  # [H,W,2*dim_each]

        # Concatenate x and y encodings → (H, W, 4*dim_each)
        emb = torch.cat([emb_x, emb_y], dim=-1)  # [H,W,4*dim_each]
        emb = emb.view(H * W, -1)  # flatten

        # Adjust to target dimension d_pos (clip or pad)
        if emb.shape[-1] > d_pos:
            emb = emb[:, :d_pos]
        elif emb.shape[-1] < d_pos:
            pad = torch.zeros(H * W, d_pos - emb.shape[-1], device=device)
            emb = torch.cat([emb, pad], dim=-1)
        return emb  # [H*W, d_pos]

    def forward_corev1(self, x_T: torch.Tensor, x_R: torch.Tensor, proximity_map):
        """
        Core forward function of DSMamba with depth-aware selective scan.

        Args:
            x_T: Transmission feature tensor [B, C, H, W].
            x_R: Reflection feature tensor [B, C, H, W].
            proximity_map: Depth (or proximity) map [B, 1, H, W].

        Returns:
            y_T: Updated transmission feature
            y_R: Updated reflection feature
        """
        B, C, H, W = x_T.shape
        L = H * W  # Flattened spatial length

        proximity_map_T = self.fuse(torch.cat([x_T, proximity_map], dim=1))          # (B, C+1, H, W)
        proximity_map_R = self.fuse(torch.cat([x_R, proximity_map], dim=1))          # (B, C+1, H, W)

        # Normalize with sigmoid, range [0,1]
        proximity_map_T = torch.sigmoid(proximity_map_T)
        proximity_map_R = torch.sigmoid(proximity_map_R)
        # ----------------------------------------------------
        # Compute scanning orders
        # ----------------------------------------------------
        # Transmission: regional scanning (large-area-first + near-to-far)
        sorted_indices_2d_T = self.Depth_Aware_Regional_Scanning(proximity_map_T, total_area=L)
        # Reflection: global scanning (all pixels sorted globally by depth)
        sorted_indices_2d_R = self.Depth_Aware_Global_Scanning(proximity_map_R)

        xs_ch_T = self.cross_scan_2d(x_T, sorted_indices_2d_T)  # [B, 2, C, L]
        xs_ch_R = self.cross_scan_2d(x_R, sorted_indices_2d_R)  # [B, 2, C, L]

        # Spatial Positional Encoding (SPE)
        pos_emb = self.Spatial_Positional_Encoding(H, W, self.d_inner, device)  # [L, d_inner]
        pe_map = pos_emb.t()[None, None, :, :]  # → [1, 1, d_inner, L]

        # broadcast for forward+backward scan
        pe_map = pe_map.expand(B, 2, self.d_inner, L)

        # Align SPE with scanning order (sorted indices)
        sorted_pos_idx_T = torch.stack([sorted_indices_2d_T, sorted_indices_2d_T.flip(dims=[1])], dim=1)  # [B, 2, L]
        sorted_pos_idx_R = torch.stack([sorted_indices_2d_R, sorted_indices_2d_R.flip(dims=[1])], dim=1)  # [B, 2, L]

        # Expand to match d_inner, then gather
        sorted_pos_idx_T = sorted_pos_idx_T.unsqueeze(2).expand(B, 2, self.d_inner, L)
        sorted_pos_idx_R = sorted_pos_idx_R.unsqueeze(2).expand(B, 2, self.d_inner, L)
        pe_T = torch.gather(pe_map, dim=3, index=sorted_pos_idx_T)
        pe_R = torch.gather(pe_map, dim=3, index=sorted_pos_idx_R)

        # Add positional encoding into scanned features
        xs_ch_T = xs_ch_T + pe_T
        xs_ch_R = xs_ch_R + pe_R

        # ----------------------------------------------------
        #  Project features into state-space params
        # ----------------------------------------------------
        # Each feature vector is projected into:
        #   - dts (time step coefficients)
        #   - B, C (state update matrices)
        x_dbl_ch_T = torch.einsum("b k d l, k c d -> b k c l",
                                xs_ch_T, self.x_proj_weight_channel)
        x_dbl_ch_R = torch.einsum("b k d l, k c d -> b k c l",
                                xs_ch_R, self.x_proj_weight_channel)

        # Depth encoding branch → [B, d_pos, L]
        depth_feat = self.depth_encoder(proximity_map).view(B, self.d_pos, L)  # [B, d_pos, L]

        sorted_indices_T = torch.stack([sorted_indices_2d_T, sorted_indices_2d_T.flip(dims=[1])], dim=1)  # [B, 2, L]
        sorted_indices_R = torch.stack([sorted_indices_2d_R, sorted_indices_2d_R.flip(dims=[1])], dim=1)  # [B, 2, L]

        # Align depth features with scanning order
        depth_proj = torch.einsum("b d l, k c d -> b k c l", depth_feat,
                                  self.depth_proj_weight_channel)  # [B, K, d_state, L]


        depth_delta_T = torch.gather(
            depth_proj, dim=3,
            index=sorted_indices_T.unsqueeze(2).expand(B, self.K,  2*self.d_state, L)
        )
        depth_delta_R = torch.gather(
            depth_proj, dim=3,
            index=sorted_indices_R.unsqueeze(2).expand(B, self.K, 2*self.d_state, L)
        )

        # Split into dt, B, C terms
        dts_im_T, B_orig_T, C_orig_T = x_dbl_ch_T.split([self.dt_rank, self.d_state, self.d_state], dim=2)
        dts_im_R, B_orig_R, C_orig_R = x_dbl_ch_R.split([self.dt_rank, self.d_state, self.d_state], dim=2)
        B_depth_T, C_depth_T = depth_delta_T.split([self.d_state, self.d_state], dim=2)
        B_depth_R, C_depth_R = depth_delta_R.split([self.d_state, self.d_state], dim=2)

        depth_sig = torch.sigmoid(self.conv_depth_sig(proximity_map))  # [B, d_state, H, W]
        depth_sig = depth_sig.view(B, self.d_state, L)  # [B, d_state, L]

        # Align depth weights with scanning order
        weight_T = torch.gather(
            depth_sig.unsqueeze(1).expand(B, 2, self.d_state, L),  # [B,2,d_state,L]
            dim=3,
            index=sorted_indices_T.unsqueeze(2).expand(B, 2, self.d_state, L)  # [B,2,d_state,L]
        )
        weight_R = torch.gather(
            depth_sig.unsqueeze(1).expand(B, 2, self.d_state, L),
            dim=3,
            index=sorted_indices_R.unsqueeze(2).expand(B, 2, self.d_state, L)
        )

        # Depth Synergized State-Space Model (DS-SSM)
        Bs_T = (1 - weight_T) * B_orig_T + weight_T * B_depth_T  # [B,K,d_state,L]
        Cs_T = (1 - weight_T) * C_orig_T + weight_T * C_depth_T
        Bs_R = (1 - weight_R) * B_orig_R + weight_R * B_depth_R  # [B,K,d_state,L]
        Cs_R = (1 - weight_R) * C_orig_R + weight_R * C_depth_R

        # ----------------------------------------------------
        #  Selective scan update
        # ----------------------------------------------------
        # dts: time-step dynamics
        dts_T = torch.einsum("b k r l, k d r -> b k d l",
                           dts_im_T, self.dt_projs_weight_channel)
        dts_T = dts_T.contiguous().view(B, -1, L)
        xs_ch_T = xs_ch_T.view(B, -1, L)

        dts_R = torch.einsum("b k r l, k d r -> b k d l",
                           dts_im_R, self.dt_projs_weight_channel)
        dts_R = dts_R.contiguous().view(B, -1, L)
        xs_ch_R = xs_ch_R.view(B, -1, L)

        dt_bias = self.dt_projs_bias_channel
        delta_bias = dt_bias.view(-1)  # [K*d_state]

        # Static state-space matrices
        As_T = -torch.exp(self.A_logs_ch_T.float())  # [1, K*d_state, 1]
        Ds_T = self.Ds_ch_T

        # Run selective scan
        out_y_ch_T = self.selective_scan(
            xs_ch_T, dts_T,
            As_T, Bs_T, Cs_T, Ds_T,
            delta_bias=delta_bias,
            delta_softplus=True,
        ).view(B, 2, -1, L)
        
        As_R = -torch.exp(self.A_logs_ch_R.float())  # [1, K*d_state, 1]
        Ds_R = self.Ds_ch_R
        out_y_ch_R = self.selective_scan(
            xs_ch_R, dts_R,
            As_R, Bs_R, Cs_R, Ds_R,
            delta_bias=delta_bias,
            delta_softplus=True,
        ).view(B, 2, -1, L)

        # ----------------------------------------------------
        #  Restore spatial structure
        # ----------------------------------------------------
        # Use inverse indices to map scanned sequence back to original 2D layout
        inverse_indices_asc_T = torch.argsort(sorted_indices_2d_T, dim=-1)  # [B, L]
        sorted_indices_desc_T = sorted_indices_2d_T.flip(dims=[1])
        inverse_indices_desc_T = torch.argsort(sorted_indices_desc_T, dim=-1)
        # 恢复正向部分
        restored_y_ch_asc_T = torch.gather(out_y_ch_T[:, 0, :, :],
                                         dim=2,
                                         index=inverse_indices_asc_T.unsqueeze(1).expand(B, out_y_ch_T.shape[2], L))
        restored_y_ch_desc_T = torch.gather(out_y_ch_T[:, 1, :, :],
                                          dim=2,
                                          index=inverse_indices_desc_T.unsqueeze(1).expand(B, out_y_ch_T.shape[2], L))
        y_T = restored_y_ch_asc_T + restored_y_ch_desc_T  # [B, C_ch, L]
        y_T = y_T.permute(0, 2, 1).view(B, H, W, self.d_inner)

        inverse_indices_asc_R = torch.argsort(sorted_indices_2d_R, dim=-1)  # [B, L]
        sorted_indices_desc_R = sorted_indices_2d_R.flip(dims=[1])
        inverse_indices_desc_R = torch.argsort(sorted_indices_desc_R, dim=-1)
        # 恢复正向部分
        restored_y_ch_asc_R = torch.gather(out_y_ch_R[:, 0, :, :],
                                         dim=2,
                                         index=inverse_indices_asc_R.unsqueeze(1).expand(B, out_y_ch_R.shape[2], L))
        restored_y_ch_desc_R = torch.gather(out_y_ch_R[:, 1, :, :],
                                          dim=2,
                                          index=inverse_indices_desc_R.unsqueeze(1).expand(B, out_y_ch_R.shape[2], L))
        y_R = restored_y_ch_asc_R + restored_y_ch_desc_R  # [B, C_ch, L]

        y_R = y_R.permute(0, 2, 1).view(B, H, W, self.d_inner)

        # ----------------------------------------------------
        #  Normalize / softmax output
        # ----------------------------------------------------
        if self.softmax_version:
            y_T = torch.softmax(y_T, dim=-1).to(x_T.dtype)
            y_R = torch.softmax(y_R, dim=-1).to(x_T.dtype)
        else:
            y_T = self.out_norm(y_T).to(x_T.dtype)
            y_R = self.out_norm(y_R).to(x_T.dtype)

        return y_T, y_R

    forward_core = forward_corev1

    def forward(self, x_T: torch.Tensor, x_R: torch.Tensor, proximity_map):
        """
        Forward pass for DS-Mamba.

        Args:
            x_T: Transmission feature tensor [B, C, H, W].
            x_R: Reflection feature tensor [B, C, H, W].
            proximity_map: proximity feature [B, C, H, W].

        Returns:
            out_T, out_R: Updated transmission and reflection features.
        """
        # (1) Input projection to expanded dimension
        x_T = x_T.permute(0, 2, 3, 1).contiguous()
        x_R = x_R.permute(0, 2, 3, 1).contiguous()
        xz_T, xz_R = self.in_proj(x_T), self.in_proj(x_R)
        x_T, z_T = xz_T.chunk(2, dim=-1)  # Split into state and gating branch
        x_R, z_R = xz_R.chunk(2, dim=-1)
        x_T = x_T.permute(0, 3, 1, 2).contiguous()
        x_R = x_R.permute(0, 3, 1, 2).contiguous()

        # (2) Depthwise conv + activation
        x_T = self.act(self.conv2d(x_T))
        x_R = self.act(self.conv2d(x_R))

        # (3) Core selective scan with depth-guided scanning orders
        y_T, y_R = self.forward_core(x_T, x_R, proximity_map)

        # (4) Modulate outputs with gating branch
        y_T = y_T * F.silu(z_T)
        y_R = y_R * F.silu(z_R)

        # (5) Project back to original dimension
        out_T = self.out_proj(y_T)
        out_R = self.out_proj(y_R)
        if self.dropout is not None:
            out_T = self.dropout(out_T)
            out_R = self.dropout(out_R)

        # (6) Restore shape to (B, C, H, W)
        out_T = out_T.permute(0, 3, 1, 2).contiguous()
        out_R = out_R.permute(0, 3, 1, 2).contiguous()
        return out_T, out_R

    def compute_scanning_order_np(self, depth_np: np.ndarray, total_area) -> np.ndarray:
        """
        Compute the scanning order indices for a single depth map (H, W).
        """
        N = total_area  # 假设 total_area == H * W

        # 1. Binarize depth map using 50th percentile as threshold
        thresh = np.percentile(depth_np, 50)
        mask = (depth_np >= thresh).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

        # 3. Compute areas of all connected regions (skip background: stats[0])
        areas = stats[1:, cv2.CC_STAT_AREA]  # shape=(num_labels-1,)
        min_area = total_area * 0.05

        # 4. Select valid regions with area >= min_area
        valid = np.where(areas >= min_area)[0] + 1
        if valid.size > 5:
            top5 = valid[np.argsort(areas[valid - 1])[::-1][:5]]
        else:
            top5 = valid

        # 5. Also include background (label 0)
        bg_label = 0
        region_labels = list(top5) + [bg_label]

        # helper: compute area of region `lab`
        def area_of(lab):
            return stats[lab, cv2.CC_STAT_AREA]

        # 6. Sort selected regions by area (descending)
        ordered_regions = sorted(region_labels, key=area_of, reverse=True)
        bg_rank = ordered_regions.index(bg_label)

        # initialize rank map with background rank
        rank_map = np.full(num_labels, fill_value=bg_rank, dtype=np.int32)
        for r, lab in enumerate(ordered_regions):
            rank_map[lab] = r

        # 7. Flatten depth map and labels
        labels_flat = labels.ravel()
        depth_flat = depth_np.ravel()
        idxs = np.arange(N, dtype=np.int64)

        # 8. Assign region rank to each pixel
        region_ranks = rank_map[labels_flat]

        # 9. Sort pixels:
        #    first by region rank (large area first),
        #    then by depth (descending)
        order = np.lexsort((-depth_flat, region_ranks))

        # 10. Return reordered indices (shape: N,)
        return idxs[order]

    def Depth_Aware_Regional_Scanning(self, depth_tensor: torch.Tensor,total_area) -> torch.Tensor:
        B, _, H, W = depth_tensor.shape  # depth_tensor: [B, 1, H, W]
        orders = []
        # Convert tensor to numpy for OpenCV ops
        depth_np = depth_tensor.squeeze(1).detach().cpu().numpy()  # [B, H, W]
        for i in range(B):
            # Compute scanning order for each depth map
            order_np = self.compute_scanning_order_np(depth_np[i],total_area)
            orders.append(order_np)
        # Stack batch results into (B, H*W)
        orders = np.stack(orders, axis=0)  # [B, H*W]
        # Convert back to tensor on same device
        orders_tensor = torch.from_numpy(orders).to(depth_tensor.device).long()
        return orders_tensor

    def Depth_Aware_Global_Scanning(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """
        Global scanning order: sort all pixels by depth (descending).
        """
        B, C, H, W = depth_tensor.shape  # depth_tensor: [B, C, H, W]

        # Flatten depth map to (B, C, H*W)
        depth_flat = depth_tensor.view(B, C, -1)  # shape: [B, C, H*W]

        # Sort indices in descending order of depth values
        sorted_indices = torch.argsort(depth_flat, dim=2, descending=True)  # shape: [B, C, H*W]

        # Remove channel dimension if C=1 → (B, H*W)
        return sorted_indices.squeeze(1)


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        """
        Initialize delta-time (dt) projection.
        """
        # Linear layer: projects rank -> d_inner
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        # Initialize weight
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize bias using exponential distribution in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        """
        Initialize log(A), the state transition matrix (diagonal form).
        """
        # Create sequence 1..d_state, repeat along d_inner
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)

        # Replicate if multiple copies needed
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)

        # Convert to learnable parameter
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # Start with all ones
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        # Convert to learnable parameter
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

class GateNetwork(nn.Module):
    def  __init__(self, input_size, num_experts, top_k):
        super(GateNetwork, self).__init__()
        # Global pooling layers to capture global context
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        # Two fully connected layers for scoring
        self.fc0 = nn.Linear(input_size,num_experts)
        self.fc1 = nn.Linear(input_size, num_experts)
        self.relu1 = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        # Initialize fc1 weights to zero to stabilize training at start
        torch.nn.init.zeros_(self.fc1.weight)
        # Softplus ensures noise is positive and smooth
        self.sp = nn.Softplus()
    def forward(self, x):
        """
        Args:
            x: Input feature map of shape [B, C, H, W].
        Returns:
            gating_coeffs: Tensor of shape [B, num_experts],
                           softmax-normalized gating weights per expert.
        """
        # Step 1. Global pooling → squeeze spatial dimension
        x = self.gap(x)+self.gap2(x)
        x = x.view(-1, self.input_size) #(batch_size, C)
        inp = x

        # Step 2. Compute raw expert scores from fc1
        x = self.fc1(x)
        x= self.relu1(x)

        # Step 3. Compute smooth noise term (exploration)
        noise = self.sp(self.fc0(inp)) #(batch_size, num_experts)

        # Normalize noise to zero mean and unit variance (per sample)
        noise_mean = torch.mean(noise,dim=1)
        noise_mean = noise_mean.view(-1,1)
        std = torch.std(noise,dim=1)
        std = std.view(-1,1)
        noram_noise = (noise-noise_mean)/std
        # Step 4. Add noise to scores and select Top-K experts
        topk_values, topk_indices = torch.topk(x+noram_noise, k=self.top_k, dim=1)

        # Step 5. Build mask for Top-K positions
        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)

        # Suppress all non-topK scores by setting them to -inf
        x[~mask.bool()] = float('-inf')

        # Step 6. Apply softmax across experts → gating distribution
        gating_coeffs = self.softmax(x)  # (batch_size, num_experts)
        # Each row sums to 1, only Top-K experts get non-zero weights

        return gating_coeffs

class Memory(nn.Module):
    def __init__(self, channel_dim, dilation):
        """
        Memory module for expert networks.
        Each memory stores prototypical features and interacts with input queries
        through both global pattern matching and spatial refinement streams.

        Args:
            channel_dim (int): Dimensionality of input feature channels (D).
            dilation (int): Dilation rate for depth-wise convolution (controls
                            receptive field expansion for spatial refinement).
        """
        super(Memory, self).__init__()
        # Fusion conv: combines outputs of global and spatial streams
        self.fusion = nn.Sequential(nn.Conv2d(2*channel_dim, channel_dim, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(0.2))

        # Depth-wise conv: enlarges receptive field while keeping channel structure
        self.depth_conv = nn.Sequential(nn.Conv2d(channel_dim, channel_dim, kernel_size=3, stride=1, padding=dilation, groups=channel_dim, dilation=dilation),nn.LeakyReLU(0.2))

        # Number of top memory slots to attend in spatial refinement
        self.topk = 2

    def forward(self, image_feature, memory, train):
        """
        Forward pass of Memory module.

        Args:
            image_feature (Tensor): Input query feature map (B, D, H, W).
            memory (Tensor): Memory bank (M, D), where M = number of slots.
            train (bool): If True, update memory and compute losses.

        Returns:
            updated_image (Tensor): Updated features after memory interaction (B, D, H, W).
            updated_memory (Tensor): Updated memory slots (M, D).
            loss_mem_align (float): Memory alignment loss (0 if train=False).
            loss_mem_triplet (float): Memory triplet loss (0 if train=False).
        """

        global_compensation, updated_memory, loss_mem_align, loss_mem_triplet = self.Global_Pattern_Interaction_Stream(image_feature, memory, train)
        memory_feat = self.Spatial_Context_Refinement_Stream(image_feature, memory)

        # 卷积融合：通过 fusion_conv 将 (B, 2*D, H, W) 转换为 (B, D, H, W)
        updated_image = self.fusion(torch.cat([global_compensation,memory_feat],dim=1))  # (B, D, H, W)
        updated_image = self.depth_conv(updated_image)  # (B, D, H, W)

        return updated_image, updated_memory,  loss_mem_align, loss_mem_triplet

    def Spatial_Context_Refinement_Stream(self, image_feature, memory):
        B, D, H, W = image_feature.shape
        M, _ = memory.shape

        # Treat memory as conv kernels: measure similarity (B, M, H, W)
        memory_kernels = memory.view(M, D, 1, 1)
        score_maps = F.conv2d(image_feature, weight=memory_kernels)  # (B, M, H, W)

        # Flatten similarity and select top-K memory slots
        score_maps_flat = score_maps.view(B, M, -1)  # (B, M, HW)
        topk_scores, topk_indices = torch.topk(score_maps_flat, k=self.topk, dim=1)  # (B, K, HW)

        # Normalize attention across top-K slots
        attn_weights = F.softmax(topk_scores, dim=1)  # (B, K, HW)

        # Gather top-K memory embeddings
        idx_flat = topk_indices.permute(0, 2, 1).reshape(B, -1)  # (B, HW*K)

        # gather: keys_expanded ∈ (1, M, D) → index ∈ (B, HW*K) → output ∈ (B, HW*K, D)
        keys_exp = memory.unsqueeze(0).expand(B, -1, -1)  # (B, M, D)
        gathered_keys = torch.gather(keys_exp, dim=1, index=idx_flat.unsqueeze(-1).expand(-1, -1, D))  # (B, HW*K, D)

        #  reshape back to (B, HW, K, D) for attention
        gathered_keys = gathered_keys.view(B, H * W, self.topk, D).permute(0, 2, 1, 3)  # (B, K, HW, D)

        # attention weights: (B, K, HW, 1)
        attn_weights = attn_weights.unsqueeze(-1)  # (B, K, HW, 1)

        # Weighted sum of memory
        memory_feat_flat = torch.sum(attn_weights * gathered_keys, dim=1)  # (B, HW, D)

        # reshape back
        memory_feat = memory_feat_flat.transpose(1, 2).reshape(B, D, H, W)
        return memory_feat


    def Global_Pattern_Interaction_Stream(self, image_feature, memory, train):
        if train:
            # Step 1: Perform global pattern adjustment between image features and memory
            I_G, global_compensation, score_memory, score_image = self.Global_Pattern_Adjustment(image_feature, memory)

            # Step 2: Update memory using the adjusted features and similarity scores
            updated_memory, gathering_indices = self.Memory_Evolution(I_G, memory, score_memory, score_image, train)

            # Step 3: Compute memory alignment loss based on updated memory and gathered indices
            loss_mem_align = self.loss_mem_align(I_G, memory, gathering_indices)

            # Step 4: Compute triplet loss to enforce memory discrimination
            loss_mem_triplet = self.loss_mem_triplet(I_G, memory, score_image)

            # Return compensated features, updated memory, and two loss terms
            return global_compensation, updated_memory, loss_mem_align, loss_mem_triplet
        else:
            # Step 1: Perform global pattern adjustment (no memory update during inference)
            I_G, global_compensation, score_memory, score_image = self.Global_Pattern_Adjustment(image_feature, memory)

            # Return compensated features, original memory, and zero losses
            return global_compensation, memory, 0, 0

    def Global_Pattern_Adjustment(self, image_feature, memory):
        """
        Args:
            image_feature: (B, D, H, W).
            memory: (M, D).

        Returns:
            I_G: Global pooled query (B, 1, 1, D).
            global_compensation: Compensated global feature (B, D, H, W).
            score_memory: Similarity scores normalized over queries (B*HW, M).
            score_image: Similarity scores normalized over memory (B*HW, M).
        """

        I_G = F.adaptive_avg_pool2d(image_feature, output_size=(1, 1))  # (B, D, 1, 1)
        I_G = I_G.permute(0, 2, 3, 1)  # (B, 1, 1, D)
        b, h, w, d = I_G.size()
        score_memory, score_image = self.get_score(memory, I_G)  # (B*h*w, M)
        I_G_flat = I_G.contiguous().view(b * h * w, d)  # (B*h*w, d)

        memory_response = torch.matmul(score_image.detach(), memory)  # (B*h*w, d)
        memory_response = I_G_flat + memory_response  # (B*h*w, d)
        memory_response = memory_response.view(b, h, w, d).permute(0, 3, 1, 2)  # (B, d, h, w)
        global_compensation = torch.sigmoid(memory_response) * image_feature

        return I_G, global_compensation, score_memory, score_image

    def get_score(self, memory, I_G):
        """
               Compute query-to-memory similarity.

               Args:
                   memory: (M, D).
                   I_G: (B, h, w, D).

               Returns:
                   score_memory: Normalized over queries (B*h*w, M).
                   score_image: Normalized over memory (B*h*w, M).
               """
        b, h, w, d = I_G.size()
        m, d = memory.size()
        score = torch.matmul(I_G, torch.t(memory))  # (B, h, w, M)
        score = score.view(b * h * w, m)  # (B*h*w, M)
        score_memory = F.softmax(score, dim=0)
        score_image = F.softmax(score, dim=1)
        return score_memory, score_image

    def Memory_Evolution(self, I_G, memory, score_memory, score_image, train):
        """
              Update memory by aggregating query features.

              Args:
                  I_G: Global query (B, 1, 1, D).
                  memory: (M, D).
                  score_memory: (B*h*w, M).
                  score_image: (B*h*w, M).

              Returns:
                  updated_memory: (M, D).
                  gathering_indices: (B*h*w, 1).
              """
        b, h, w, d = I_G.size()
        I_G_flat = I_G.contiguous().view(b * h * w, d)  # (B*h*w, d)
        _, gathering_indices = torch.topk(score_image, 1, dim=1)  # (B*h*w, 1)
        weights = score_memory.gather(1, gathering_indices)  # (N, 1)
        update_vector = I_G_flat * weights  # (N, d)
        N, d = I_G_flat.shape

        # Aggregate into memory slots
        M = score_memory.size(1)
        output = torch.zeros((M, d), device=I_G_flat.device, dtype=I_G_flat.dtype)
        gathering_indices_exp = gathering_indices.expand(-1, d)  # (N, d)
        memory_increment = output.scatter_add_(0, gathering_indices_exp, update_vector)  # 聚合到 (M, d)
        updated_memory = F.normalize(memory_increment + memory, dim=1)  # (M, d)
        return updated_memory.detach(), gathering_indices

    def loss_mem_triplet(self, I_G, memory, score_image):
        """
        Triplet loss: encourage query closer to best-matching memory
        and farther from second-best.

        Args:
            I_G: (B, 1, 1, D).
            memory: (M, D).
            score_image: (B*h*w, M).

        Returns:
            loss: scalar.
        """
        b, h, w, d = I_G.size()
        if score_image.size(1) < 2:
            # If the number of memory items is insufficient, directly return zero loss
            return torch.tensor(0.0, device=I_G.device)
        loss_fn = nn.TripletMarginLoss(margin=1.0)
        I_G_flat = I_G.contiguous().view(b * h * w, d)  # (B*h*w, d)
        _, gathering_indices = torch.topk(score_image, 2, dim=1)  # (B*h*w, 2)
        pos = memory[gathering_indices[:, 0]]  # (B*h*w, d)
        neg = memory[gathering_indices[:, 1]]  # (B*h*w, d)
        loss = loss_fn(I_G_flat, pos.detach(), neg.detach())
        return loss

    def loss_mem_align(self, I_G, memory, gathering_indices):
        """
              Alignment loss: encourage query and its closest memory to align.

              Args:
                  I_G: (B, 1, 1, D).
                  memory: (M, D).
                  gathering_indices: (B*h*w, 1).

              Returns:
                  loss: scalar.
              """
        b, h, w, d = I_G.size()
        loss_fn = nn.MSELoss()
        I_G_flat = I_G.contiguous().view(b * h * w, d)  # (B*h*w, d)
        loss = loss_fn(I_G_flat, memory[gathering_indices].squeeze(1).detach())
        return loss


class Expert(nn.Module):
    def __init__(self, memory_module):
        super(Expert, self).__init__()
        # Each expert has its own memory module
        # memory_module should implement the forward method:
        #     (x, memory, train) → (updated_x, updated_memory, loss_align, loss_triplet)
        self.memory_module = memory_module

    def forward(self, x, memory,train):
        """
        Args:
            x: Input features for this expert. Shape: [B_expert, C, H, W],
               where B_expert is the number of samples routed to this expert.
            memory: The expert’s private memory bank (tensor or dict).
            train: Boolean flag. If True, compute memory alignment and triplet losses.

        Returns:
            out: Processed features after memory interaction. Shape same as x.
            updated_memory: Updated memory bank for this expert.
            loss_mem_align: Memory alignment loss (encourages feature consistency).
            loss_mem_triplet: Triplet loss for discriminative memory usage.
        """
        # Each expert processes input using its own memory module
        out, updated_memory, loss_mem_align, loss_mem_triplet = self.memory_module(x, memory,train)
        return out, updated_memory, loss_mem_align, loss_mem_triplet

class Memory_Expert_Compensation_Module(nn.Module):
    def __init__(self, channels, num_experts, top_k):
        super(Memory_Expert_Compensation_Module, self).__init__()
        # Gate network generates soft assignment (coefficients) for selecting experts
        self.gate = GateNetwork(channels, num_experts, top_k)

        # Each expert receives a unique dilation factor to diversify receptive fields
        dilations = [2 * i + 1 for i in range(num_experts)]
        assert num_experts == len(dilations)

        # Construct a pool of experts. Each expert has its own memory module.
        # Memory module is initialized with the corresponding dilation factor.
        self.Memory_Experts = nn.ModuleList([
            Expert(Memory(channels, dilation=d)) for d in dilations
        ])

    def forward(self, x, memory_list,train):
        """
               Forward pass of the Memory Expert Compensation Module (MECM).

               Args:
                   x: Input features [B, C, H, W].
                   memory_list: List of expert memories, shape [num_experts, (C, C)].
                   train: Boolean flag indicating whether in training mode.

               Returns:
                   out: Expert-augmented features [B, C, H, W].
                   updated_memories: List of updated memory items for each expert.
                   loss_mem_aligns: Memory alignment loss (averaged across experts).
                   loss_mem_triplets: Memory triplet loss (averaged across experts).
                   cof: Gating coefficients [B, num_experts].
               """
        # Step 1. Compute gating coefficients for each expert
        cof = self.gate(x) # (batch_size, num_experts)

        # Step 2. Initialize output feature (same shape as input)
        out = torch.zeros_like(x).to(x.device)

        # Step 3. Prepare containers for updated memories and memory losses
        updated_memories = []
        loss_mem_aligns = []
        loss_mem_triplets = []
        # Step 4. Iterate through all experts
        for idx in range(len(self.Memory_Experts)):
            # If no sample in the batch selects this expert → keep memory unchanged
            if cof[:,idx].all()==0:
                updated_memories.append(memory_list[idx])  # 插入原始的记忆而非跳过
                continue
            # Otherwise, find batch indices where this expert is selected
            mask = torch.where(cof[:, idx] > 0)[0]
            # Only forward those samples through the expert

            # Each expert processes its own subset of data and updates its memory
            expert_out,  updated_memory, loss_mem_align, loss_mem_triplet = self.Memory_Experts[idx](x[mask], memory_list[idx],train)
            # Save updated memory and losses
            updated_memories.append(updated_memory)
            loss_mem_aligns.append(loss_mem_align)
            loss_mem_triplets.append(loss_mem_triplet)

            # Step 5. Apply expert contribution to the output, scaled by gate coefficient
            cof_k = cof[mask,idx].view(-1, 1, 1, 1)
            out[mask] += expert_out * cof_k
        # Step 6. Aggregate losses across all experts
        if train:
            # Stack all experts’ losses and average
            loss_mem_aligns = torch.stack(loss_mem_aligns).mean()
            loss_mem_triplets = torch.stack(loss_mem_triplets).mean()
        else:
            loss_mem_aligns = 0
            loss_mem_triplets = 0

        return out, updated_memories, loss_mem_aligns, loss_mem_triplets,cof


class Depth_Memory_Decoupling_Block(nn.Module):
    def __init__(self, dim,is_last_block, num_experts=settings.num_experts,top_experts=settings.top_experts,ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        """
               Depth-Memory Decoupling Block (DMDBlock).

               Args:
                   dim: Feature dimension at this level.
                   is_last_block: Boolean flag indicating if this is the last block in MultiInputSequential.
                                  Only the last block performs memory expert interaction (MECM).
                   num_experts: Number of memory experts in MECM.
                   top_experts: Number of top-k experts to be selected in MECM.
                   ffn_expansion_factor: Expansion ratio for FeedForward network.
                   bias: Whether to use bias in convolution/linear layers.
                   LayerNorm_type: Type of LayerNorm (with or without bias).
               """
        super(Depth_Memory_Decoupling_Block, self).__init__()
        self.dim = dim
        self.is_last_block = is_last_block  # 记录当前 Block 的索引
        self.norm1 = LayerNorm(self.dim, LayerNorm_type)
        # Depth-Synergized Decoupling Mamba
        self.DSMamba = Depth_Synergized_Decoupling_Mamba(d_model=self.dim, ssm_ratio=1)
        self.norm2 = LayerNorm(self.dim, LayerNorm_type)
        self.EFFN = FeedForward(self.dim, ffn_expansion_factor, bias)
        if self.is_last_block:
            self.MECM = Memory_Expert_Compensation_Module(channels=self.dim, num_experts=num_experts, top_k=top_experts)

    def forward(self, xT, xR, depth_T, memory_T, memory_R, train):
        xT_ori = xT
        xR_ori = xR

        xT, xR = self.DSMamba(self.norm1(xT_ori), self.norm1(xR_ori), depth_T)
        xT, xR = xT + xT_ori, xR + xR_ori
        xT = xT + self.EFFN(self.norm2(xT))
        xR = xR + self.EFFN(self.norm2(xR))

        if self.is_last_block:
            MoE_T, memory_T, loss_mem_align_T, loss_mem_triplet_T, cof_T = self.MECM(xT, memory_T, train)
            MoE_R, memory_R, loss_mem_align_R, loss_mem_triplet_R, cof_R = self.MECM(xR, memory_R, train)
            xT = MoE_T + xT
            xR = MoE_R + xR
            return xT, xR, memory_T, loss_mem_align_T, loss_mem_triplet_T, cof_T, memory_R, loss_mem_align_R, loss_mem_triplet_R, cof_R
        else:
            return xT, xR


class MultiInputSequential(nn.Module):
    def __init__(self, *blocks):
        """
        A sequential container that supports multiple inputs (xT, xR, depth, memory_T, memory_R).
        Unlike nn.Sequential, this class allows each block to handle transmission (T) and reflection (R) streams,
        as well as depth and memory modules.

        Args:
            *blocks: A list of blocks (Depth_Memory_Decoupling_Block).
        """
        super(MultiInputSequential, self).__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, xT, xR, depth_T, memory_T, memory_R, train):
        # Initialize accumulators for memory losses
        total_loss_mem_align_T, total_loss_mem_triplet_T = 0, 0
        total_loss_mem_align_R, total_loss_mem_triplet_R = 0, 0
        # Lists to store gating coefficients (from the last block only)
        cof_T_list = []
        cof_R_list = []
        num_blocks = len(self.blocks)

        for i, block in enumerate(self.blocks):
            is_last_block = (i == num_blocks - 1)
            if is_last_block:
                xT, xR, memory_T, loss_mem_align_T, loss_mem_triplet_T, cof_T, memory_R, loss_mem_align_R, loss_mem_triplet_R, cof_R = block(
                    xT, xR, depth_T, memory_T, memory_R, train)
                # Store the memory losses from the last block
                total_loss_mem_align_T = loss_mem_align_T
                total_loss_mem_triplet_T = loss_mem_triplet_T
                total_loss_mem_align_R = loss_mem_align_R
                total_loss_mem_triplet_R = loss_mem_triplet_R

                cof_T_list.append(cof_T)
                cof_R_list.append(cof_R)
            else:
                # Intermediate blocks: only update features (no memory update, no loss computation)
                xT, xR = block(xT, xR, depth_T, memory_T, memory_R, train)
        return xT, xR, memory_T, total_loss_mem_align_T, total_loss_mem_triplet_T, cof_T_list, memory_R, total_loss_mem_align_R, total_loss_mem_triplet_R, cof_R_list

MEMORY_KEYS = [
    # level1
    "memory_level1_T", "memory_level1_R",

    # level2
    "memory_level2_T", "memory_level2_R",

    # level3
    "memory_level3_T", "memory_level3_R",
    "memory_DS3_T1", "memory_DS3_T2",
    "memory_aib2_Rs", "memory_aib2_Rc",

    # level4
    "memory_level4_T", "memory_level4_R",
    "memory_DS4_T1", "memory_DS4_T2",
    "memory_aib3_Rs", "memory_aib3_Rc",

    # level5
    "memory_level5_T", "memory_level5_R",
    "memory_DS5_T1", "memory_DS5_T2",
    "memory_DS5_R1", "memory_DS5_R2",
]


class Depth_Memory_Decoupling_Network(nn.Module):
    def __init__(self,dims=settings.dims,enc_blk_nums=[4, 4, 2, 2, 1], dec_blk_nums=[2, 2, 2, 2, 1]):
        super().__init__()
        """
        Depth-Memory Decoupling Network (DMDNet).

        Args:
            dims: list of feature dimensions at each level (5 scales).
            enc_blk_nums: number of Depth-Memory decoupling blocks for each encoder level.
            dec_blk_nums: number of MuGI (dual-stream) blocks for each decoder level.
        """
        dim1, dim2, dim3, dim4, dim5 = dims

        self.Encoding_Branch = Encoding_Branch(dims, [2, 2, 2, 2, 2])

        self.DMBlock_DSBranch_level5 = MultiInputSequential(
            *[Depth_Memory_Decoupling_Block(dim=dim5, is_last_block=(i == 0)) for i in range(1)]
        )

        self.DMBlock_level5 = MultiInputSequential(
            *[Depth_Memory_Decoupling_Block(dim=dim5, is_last_block=(i == enc_blk_nums[0] - 1)) for i in range(enc_blk_nums[0])]
        )
        self.MuGIBlock_level4 = DualStreamSeq(*[MuGIBlock(dim4) for _ in range(dec_blk_nums[0])])

        self.DMBlock_DSBranch_level4 = MultiInputSequential(
            *[Depth_Memory_Decoupling_Block(dim=dim4, is_last_block=(i == 0)) for i in range(1)]
        )

        self.DMBlock_level4 = MultiInputSequential(
            *[Depth_Memory_Decoupling_Block(dim=dim4, is_last_block=(i == enc_blk_nums[1] - 1)) for i in range(enc_blk_nums[1])]
        )
        self.MuGIBlock_level3 = DualStreamSeq(*[MuGIBlock(dim3) for _ in range(dec_blk_nums[1])])


        self.DMBlock_DSBranch_level3 = MultiInputSequential(
            *[Depth_Memory_Decoupling_Block(dim=dim3, is_last_block=(i == 0)) for i in range(1)]
        )

        self.DMBlock_level3 = MultiInputSequential(
            *[Depth_Memory_Decoupling_Block(dim=dim3, is_last_block=(i == enc_blk_nums[2] - 1)) for i in range(enc_blk_nums[2])]
        )
        self.MuGIBlock_level2 = DualStreamSeq(*[MuGIBlock(dim2) for _ in range(dec_blk_nums[2])])


        self.DMBlock_level2 = MultiInputSequential(
            *[Depth_Memory_Decoupling_Block(dim=dim2, is_last_block=(i == enc_blk_nums[3] - 1)) for i in range(enc_blk_nums[3])]
        )
        self.MuGIBlock_level1 = DualStreamSeq(*[MuGIBlock(dim1) for _ in range(dec_blk_nums[3])])

        self.DMBlock_level1 = MultiInputSequential(
            *[Depth_Memory_Decoupling_Block(dim=dim1, is_last_block=(i == enc_blk_nums[4] - 1)) for i in range(enc_blk_nums[4])]
        )
        self.MuGIBlock_last = DualStreamSeq(*[MuGIBlock(dim1) for _ in range(dec_blk_nums[4])])

        self.out = DualStreamBlock(nn.Conv2d(in_channels=dim1, out_channels=3, kernel_size=3, padding=1), nn.LeakyReLU(0.2))

        # Down-projection (feature compression across levels)
        self.conv_level1_d = nn.Sequential(nn.Conv2d(dim1, dim2, 1, 1), nn.LeakyReLU(0.2))
        self.conv_level2_d = nn.Sequential(nn.Conv2d(dim2, dim3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_level3_d = nn.Sequential(nn.Conv2d(dim3, dim4, 1, 1), nn.LeakyReLU(0.2))
        self.conv_level4_d = nn.Sequential(nn.Conv2d(dim4, dim5, 1, 1), nn.LeakyReLU(0.2))

        # Up-projection (feature expansion across levels)
        self.conv_4 = nn.Sequential(nn.Conv2d(dim5, dim4, 1, 1), nn.LeakyReLU(0.2))
        self.conv_3 = nn.Sequential(nn.Conv2d(dim4, dim3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_2 = nn.Sequential(nn.Conv2d(dim3, dim2, 1, 1), nn.LeakyReLU(0.2))
        self.conv_1 = nn.Sequential(nn.Conv2d(dim2, dim1, 1, 1), nn.LeakyReLU(0.2))

        self.conv_dim3 = nn.Sequential(nn.Conv2d(96, dim3, kernel_size=1), nn.LeakyReLU(0.2))
        self.conv_dim4 = nn.Sequential(nn.Conv2d(256, dim4, kernel_size=1), nn.LeakyReLU(0.2))
        self.conv_dim5 = nn.Sequential(nn.Conv2d(512, dim5, kernel_size=1), nn.LeakyReLU(0.2))

        self.convert_d = nn.Sequential(nn.Conv2d(1, dim1, 3, 1, 1), nn.LeakyReLU(0.2))
        # Pooling and upsampling for feature resizing
        self.down = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input, depth_features_dict, memory_banks, train):
        """
        Forward pass for DMDNet (memory unified into memory_banks dict).
        Returns:
            out_T, out_R,
            loss_mem_align_T, loss_mem_align_R,
            loss_mem_triplet_T, loss_mem_triplet_R,
            cof_T_list, cof_R_list,
            memory_banks (updated)
        """
        mem = memory_banks  # ⭐ alias: no unpack / no pack

        inp, ori_size = self.check_image_size(input)

        # Depth Semantic Modulation Branch (DSBranch)
        depth_semantic_3 = depth_features_dict["layer1"]
        depth_semantic_4 = depth_features_dict["layer2"]
        depth_semantic_5 = depth_features_dict["layer3"]
        depth_semantic_3 = self.conv_dim3(depth_semantic_3)
        depth_semantic_4 = self.conv_dim4(depth_semantic_4)
        depth_semantic_5 = self.conv_dim5(depth_semantic_5)

        # Loss accumulators + coefficient lists
        total_loss_mem_align_Ts = total_loss_mem_triplet_Ts = 0
        total_loss_mem_align_Tc = total_loss_mem_triplet_Tc = 0
        total_loss_mem_align_T = total_loss_mem_triplet_T = 0

        total_loss_mem_align_Rs = total_loss_mem_triplet_Rs = 0
        total_loss_mem_align_Rc = total_loss_mem_triplet_Rc = 0
        total_loss_mem_align_R = total_loss_mem_triplet_R = 0

        num_layers_Ts = num_layers_Tc = num_layers_T = 0
        num_layers_Rs = num_layers_Rc = num_layers_R = 0

        cof_T1_list, cof_T2_list, cof_T_list = [], [], []
        cof_R1_list, cof_R2_list, cof_R_list = [], [], []

        # Build proximity maps from depth
        depth_input = depth_features_dict["final_output"]
        proximity_map_1 = self.convert_d(depth_input)
        proximity_map_2 = self.conv_level1_d(self.down(proximity_map_1))
        proximity_map_3 = self.conv_level2_d(self.down(proximity_map_2))
        proximity_map_4 = self.conv_level3_d(self.down(proximity_map_3))
        proximity_map_5 = self.conv_level4_d(self.down(proximity_map_4))

        # Encoding Branch
        (encoder_T_1, encoder_R_1), (encoder_T_2, encoder_R_2), (encoder_T_3, encoder_R_3), \
        (encoder_T_4, encoder_R_4), (encoder_T_5, encoder_R_5) = self.Encoding_Branch(inp)

        # ---------------- level 5 ----------------
        encoder_T1_5, encoder_T2_5, memory_banks["memory_DS5_T1"], loss_mem_align_Ts, loss_mem_triplet_Ts, cof_T1, \
        memory_banks["memory_DS5_T2"], loss_mem_align_Tc, loss_mem_triplet_Tc, cof_T2 = \
            self.DMBlock_DSBranch_level5(
                depth_semantic_5, encoder_T_5, proximity_map_5,
                memory_banks["memory_DS5_T1"], memory_banks["memory_DS5_T2"], train
            )

        if train:
            total_loss_mem_align_Ts += loss_mem_align_Ts
            total_loss_mem_triplet_Ts += loss_mem_triplet_Ts
            total_loss_mem_align_Tc += loss_mem_align_Tc
            total_loss_mem_triplet_Tc += loss_mem_triplet_Tc
            cof_T1_list.append(cof_T1);
            cof_T2_list.append(cof_T2)
            num_layers_Ts += 1;
            num_layers_Tc += 1

        encoder_R1_5, encoder_R2_5, memory_banks["memory_DS5_R1"], loss_mem_align_Rs, loss_mem_triplet_Rs, cof_R1, \
        memory_banks["memory_DS5_R2"], loss_mem_align_Rc, loss_mem_triplet_Rc, cof_R2 = \
            self.DMBlock_DSBranch_level5(
                depth_semantic_5, encoder_R_5, proximity_map_5,
                memory_banks["memory_DS5_R1"], memory_banks["memory_DS5_R2"], train
            )

        if train:
            total_loss_mem_align_Rs += loss_mem_align_Rs
            total_loss_mem_triplet_Rs += loss_mem_triplet_Rs
            total_loss_mem_align_Rc += loss_mem_align_Rc
            total_loss_mem_triplet_Rc += loss_mem_triplet_Rc
            cof_R1_list.append(cof_R1);
            cof_R2_list.append(cof_R2)
            num_layers_Rs += 1;
            num_layers_Rc += 1

        decoder_T_5, decoder_R_5, memory_banks["memory_level5_T"], loss_mem_align_T, loss_mem_triplet_T, cof_T, \
        memory_banks["memory_level5_R"], loss_mem_align_R, loss_mem_triplet_R, cof_R = \
            self.DMBlock_level5(
                encoder_T1_5 + encoder_T2_5,
                encoder_R1_5 + encoder_R2_5,
                proximity_map_5,
                memory_banks["memory_level5_T"], memory_banks["memory_level5_R"],
                train
            )

        if train:
            total_loss_mem_align_T += loss_mem_align_T
            total_loss_mem_triplet_T += loss_mem_triplet_T
            total_loss_mem_align_R += loss_mem_align_R
            total_loss_mem_triplet_R += loss_mem_triplet_R
            cof_T_list.append(cof_T);
            cof_R_list.append(cof_R)
            num_layers_T += 1;
            num_layers_R += 1

        decoder_T_5 = self.conv_4(self.up(decoder_T_5))
        decoder_R_5 = self.conv_4(self.up(decoder_R_5))
        proximity_map_5_up = self.conv_4(self.up(proximity_map_5))
        decoder_T_5, decoder_R_5 = self.MuGIBlock_level4(decoder_T_5, decoder_R_5)

        # ---------------- level 4 ----------------
        encoder_T1_4, encoder_T2_4, memory_banks["memory_DS4_T1"], loss_mem_align_Ts, loss_mem_triplet_Ts, cof_T1, \
        memory_banks["memory_DS4_T2"], loss_mem_align_Tc, loss_mem_triplet_Tc, cof_T2 = \
            self.DMBlock_DSBranch_level4(
                depth_semantic_4, encoder_T_4, proximity_map_4,
                memory_banks["memory_DS4_T1"], memory_banks["memory_DS4_T2"], train
            )

        if train:
            total_loss_mem_align_Ts += loss_mem_align_Ts
            total_loss_mem_triplet_Ts += loss_mem_triplet_Ts
            total_loss_mem_align_Tc += loss_mem_align_Tc
            total_loss_mem_triplet_Tc += loss_mem_triplet_Tc
            cof_T1_list.append(cof_T1);
            cof_T2_list.append(cof_T2)
            num_layers_Ts += 1;
            num_layers_Tc += 1

        encoder_R1_4, encoder_R2_4, memory_banks["memory_DS4_R1"], loss_mem_align_Rs, loss_mem_triplet_Rs, cof_R1, \
        memory_banks["memory_DS4_R2"], loss_mem_align_Rc, loss_mem_triplet_Rc, cof_R2 = \
            self.DMBlock_DSBranch_level4(
                depth_semantic_4, encoder_R_4, proximity_map_4,
                memory_banks["memory_DS4_R1"], memory_banks["memory_DS4_R2"], train
            )

        if train:
            total_loss_mem_align_Rs += loss_mem_align_Rs
            total_loss_mem_triplet_Rs += loss_mem_triplet_Rs
            total_loss_mem_align_Rc += loss_mem_align_Rc
            total_loss_mem_triplet_Rc += loss_mem_triplet_Rc
            cof_R1_list.append(cof_R1);
            cof_R2_list.append(cof_R2)
            num_layers_Rs += 1;
            num_layers_Rc += 1

        proximity_map_4 = proximity_map_5_up + proximity_map_4
        decoder_T_4, decoder_R_4, memory_banks["memory_level4_T"], loss_mem_align_T, loss_mem_triplet_T, cof_T, \
        memory_banks["memory_level4_R"], loss_mem_align_R, loss_mem_triplet_R, cof_R = \
            self.DMBlock_level4(
                decoder_T_5 + encoder_T1_4 + encoder_T2_4,
                decoder_R_5 + encoder_R1_4 + encoder_R2_4,
                proximity_map_4,
                memory_banks["memory_level4_T"], memory_banks["memory_level4_R"],
                train
            )

        if train:
            total_loss_mem_align_T += loss_mem_align_T
            total_loss_mem_triplet_T += loss_mem_triplet_T
            total_loss_mem_align_R += loss_mem_align_R
            total_loss_mem_triplet_R += loss_mem_triplet_R
            cof_T_list.append(cof_T);
            cof_R_list.append(cof_R)
            num_layers_T += 1;
            num_layers_R += 1

        decoder_T_4 = self.conv_3(self.up(decoder_T_4))
        decoder_R_4 = self.conv_3(self.up(decoder_R_4))
        proximity_map_4_up = self.conv_3(self.up(proximity_map_4))
        decoder_T_4, decoder_R_4 = self.MuGIBlock_level3(decoder_T_4, decoder_R_4)

        # ---------------- level 3 ----------------
        encoder_T1_3, encoder_T2_3, memory_banks["memory_DS3_T1"], loss_mem_align_Ts, loss_mem_triplet_Ts, cof_T1, \
        memory_banks["memory_DS3_T2"], loss_mem_align_Tc, loss_mem_triplet_Tc, cof_T2 = \
            self.DMBlock_DSBranch_level3(
                depth_semantic_3, encoder_T_3, proximity_map_3,
                memory_banks["memory_DS3_T1"], memory_banks["memory_DS3_T2"], train
            )

        if train:
            total_loss_mem_align_Ts += loss_mem_align_Ts
            total_loss_mem_triplet_Ts += loss_mem_triplet_Ts
            total_loss_mem_align_Tc += loss_mem_align_Tc
            total_loss_mem_triplet_Tc += loss_mem_triplet_Tc
            cof_T1_list.append(cof_T1);
            cof_T2_list.append(cof_T2)
            num_layers_Ts += 1;
            num_layers_Tc += 1

        encoder_R1_3, encoder_R2_3, memory_banks["memory_DS3_R1"], loss_mem_align_Rs, loss_mem_triplet_Rs, cof_R1, \
        memory_banks["memory_DS3_R2"], loss_mem_align_Rc, loss_mem_triplet_Rc, cof_R2 = \
            self.DMBlock_DSBranch_level3(
                depth_semantic_3, encoder_R_3, proximity_map_3,
                memory_banks["memory_DS3_R1"], memory_banks["memory_DS3_R2"], train
            )

        if train:
            total_loss_mem_align_Rs += loss_mem_align_Rs
            total_loss_mem_triplet_Rs += loss_mem_triplet_Rs
            total_loss_mem_align_Rc += loss_mem_align_Rc
            total_loss_mem_triplet_Rc += loss_mem_triplet_Rc
            cof_R1_list.append(cof_R1);
            cof_R2_list.append(cof_R2)
            num_layers_Rs += 1;
            num_layers_Rc += 1

        proximity_map_3 = proximity_map_4_up + proximity_map_3
        decoder_T_3, decoder_R_3, memory_banks["memory_level3_T"], loss_mem_align_T, loss_mem_triplet_T, cof_T, \
        memory_banks["memory_level3_R"], loss_mem_align_R, loss_mem_triplet_R, cof_R = \
            self.DMBlock_level3(
                decoder_T_4 + encoder_T1_3 + encoder_T2_3,
                decoder_R_4 + encoder_R1_3 + encoder_R2_3,
                proximity_map_3,
                memory_banks["memory_level3_T"], memory_banks["memory_level3_R"],
                train
            )

        if train:
            total_loss_mem_align_T += loss_mem_align_T
            total_loss_mem_triplet_T += loss_mem_triplet_T
            total_loss_mem_align_R += loss_mem_align_R
            total_loss_mem_triplet_R += loss_mem_triplet_R
            cof_T_list.append(cof_T);
            cof_R_list.append(cof_R)
            num_layers_T += 1;
            num_layers_R += 1

        decoder_T_3 = self.conv_2(self.up(decoder_T_3))
        decoder_R_3 = self.conv_2(self.up(decoder_R_3))
        proximity_map_3_up = self.conv_2(self.up(proximity_map_3))
        decoder_T_3, decoder_R_3 = self.MuGIBlock_level2(decoder_T_3, decoder_R_3)

        # ---------------- level 2 ----------------
        proximity_map_2 = proximity_map_3_up + proximity_map_2
        decoder_T_2, decoder_R_2, memory_banks["memory_level2_T"], loss_mem_align_T, loss_mem_triplet_T, cof_T, \
        memory_banks["memory_level2_R"], loss_mem_align_R, loss_mem_triplet_R, cof_R = \
            self.DMBlock_level2(
                decoder_T_3 + encoder_T_2,
                decoder_R_3 + encoder_R_2,
                proximity_map_2,
                memory_banks["memory_level2_T"], memory_banks["memory_level2_R"],
                train
            )

        if train:
            total_loss_mem_align_T += loss_mem_align_T
            total_loss_mem_triplet_T += loss_mem_triplet_T
            total_loss_mem_align_R += loss_mem_align_R
            total_loss_mem_triplet_R += loss_mem_triplet_R
            cof_T_list.append(cof_T);
            cof_R_list.append(cof_R)
            num_layers_T += 1;
            num_layers_R += 1

        decoder_T_2 = self.conv_1(self.up(decoder_T_2))
        decoder_R_2 = self.conv_1(self.up(decoder_R_2))
        proximity_map_2_up = self.conv_1(self.up(proximity_map_2))
        decoder_T_2, decoder_R_2 = self.MuGIBlock_level1(decoder_T_2, decoder_R_2)

        # ---------------- level 1 ----------------
        proximity_map_1 = proximity_map_2_up + proximity_map_1
        decoder_T_1, decoder_R_1, memory_banks["memory_level1_T"], loss_mem_align_T, loss_mem_triplet_T, cof_T, \
        memory_banks["memory_level1_R"], loss_mem_align_R, loss_mem_triplet_R, cof_R = \
            self.DMBlock_level1(
                decoder_T_2 + encoder_T_1,
                decoder_R_2 + encoder_R_1,
                proximity_map_1,
                memory_banks["memory_level1_T"], memory_banks["memory_level1_R"],
                train
            )

        if train:
            total_loss_mem_align_T += loss_mem_align_T
            total_loss_mem_triplet_T += loss_mem_triplet_T
            total_loss_mem_align_R += loss_mem_align_R
            total_loss_mem_triplet_R += loss_mem_triplet_R
            cof_T_list.append(cof_T);
            cof_R_list.append(cof_R)
            num_layers_T += 1;
            num_layers_R += 1

        decoder_T_1, decoder_R_1 = self.MuGIBlock_last(decoder_T_1, decoder_R_1)

        out_T, out_R = self.out(decoder_T_1, decoder_R_1)
        out_T = self.restore_image_size(out_T, ori_size)
        out_R = self.restore_image_size(out_R, ori_size)

        if train:
            # average per group
            avg_loss_mem_align_T = total_loss_mem_align_T / max(num_layers_T, 1)
            avg_loss_mem_triplet_T = total_loss_mem_triplet_T / max(num_layers_T, 1)
            avg_loss_mem_align_R = total_loss_mem_align_R / max(num_layers_R, 1)
            avg_loss_mem_triplet_R = total_loss_mem_triplet_R / max(num_layers_R, 1)

            avg_loss_mem_align_Ts = total_loss_mem_align_Ts / max(num_layers_Ts, 1)
            avg_loss_mem_triplet_Ts = total_loss_mem_triplet_Ts / max(num_layers_Ts, 1)
            avg_loss_mem_align_Tc = total_loss_mem_align_Tc / max(num_layers_Tc, 1)
            avg_loss_mem_triplet_Tc = total_loss_mem_triplet_Tc / max(num_layers_Tc, 1)

            avg_loss_mem_align_Rs = total_loss_mem_align_Rs / max(num_layers_Rs, 1)
            avg_loss_mem_triplet_Rs = total_loss_mem_triplet_Rs / max(num_layers_Rs, 1)
            avg_loss_mem_align_Rc = total_loss_mem_align_Rc / max(num_layers_Rc, 1)
            avg_loss_mem_triplet_Rc = total_loss_mem_triplet_Rc / max(num_layers_Rc, 1)

            loss_mem_align_T = (avg_loss_mem_align_T + avg_loss_mem_align_Ts + avg_loss_mem_align_Tc) / 3
            loss_mem_align_R = (avg_loss_mem_align_R + avg_loss_mem_align_Rs + avg_loss_mem_align_Rc) / 3
            loss_mem_triplet_T = (avg_loss_mem_triplet_T + avg_loss_mem_triplet_Ts + avg_loss_mem_triplet_Tc) / 3
            loss_mem_triplet_R = (avg_loss_mem_triplet_R + avg_loss_mem_triplet_Rs + avg_loss_mem_triplet_Rc) / 3

            cof_T_list.extend(cof_T1_list);
            cof_T_list.extend(cof_T2_list)
            cof_R_list.extend(cof_R1_list);
            cof_R_list.extend(cof_R2_list)
        else:
            loss_mem_align_T = loss_mem_align_R = 0
            loss_mem_triplet_T = loss_mem_triplet_R = 0
            cof_T_list, cof_R_list = [], []

        return out_T, out_R, loss_mem_align_T, loss_mem_align_R, loss_mem_triplet_T, loss_mem_triplet_R, cof_T_list, cof_R_list, memory_banks

    def check_image_size(self, x):
        _, _, h, w = x.size()
        size = settings.check_size
        mod_pad_h = (size - h % size) % size
        mod_pad_w = (size - w % size) % size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        ori_size = [h, w]
        return x, ori_size

    def restore_image_size(self, x, ori_size):
        return x[:, :, :ori_size[0], :ori_size[1]]









