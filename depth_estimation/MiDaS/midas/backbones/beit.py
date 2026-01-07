import timm
import torch
import types

import numpy as np
import torch.nn.functional as F

from .utils import forward_adapted_unflatten, make_backbone_default
from timm.models.beit import gen_relative_position_index
from torch.utils.checkpoint import checkpoint
from typing import Optional


def forward_beit(pretrained, x):
    return forward_adapted_unflatten(pretrained, x, "forward_features")


def patch_embed_forward(self, x):
    """
    Modification of timm.models.layers.patch_embed.py: PatchEmbed.forward to support arbitrary window sizes.
    """
    x = self.proj(x)
    if self.flatten:
        x = x.flatten(2).transpose(1, 2)
    x = self.norm(x)
    return x

# 2025-10-8 02:50:59

def _to_int(x):
    # 兼容 numpy / torch 标量
    if hasattr(x, "item"):
        return int(x.item())
    if isinstance(x, (np.integer,)):
        return int(x)
    return int(x)

def _get_rel_pos_bias(self, window_size):
    """
    修改自 timm.models.beit.Attention._get_rel_pos_bias，支持任意 window size。
    修复点：
      1) 将 numpy 标量转为 Python int
      2) 修正 old_sub_table 的 H/W 次序与 interpolate 的 size=(H, W) 一致
      3) 显式 align_corners=False
    """
    # 原始/新窗口尺寸（先算 Python int）
    old_h = _to_int(2 * self.window_size[0] - 1)
    old_w = _to_int(2 * self.window_size[1] - 1)
    new_h = _to_int(2 * window_size[0] - 1)
    new_w = _to_int(2 * window_size[1] - 1)

    old_table = self.relative_position_bias_table
    old_num = self.num_relative_distance
    new_num = new_h * new_w + 3

    # 只取去掉[CLS]相关的子表
    old_sub = old_table[:old_num - 3]

    # 注意这里按 (H, W) 排：reshape -> (1, H, W, C) 再 permute 到 (1, C, H, W)
    old_sub = old_sub.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2)

    # 这里的 size=(H, W)，用 Python int，显式 align_corners=False
    new_sub = F.interpolate(old_sub, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # 再还原回 (N=1, H, W, C) → (HW, C)
    new_sub = new_sub.permute(0, 2, 3, 1).reshape(new_num - 3, -1)

    # 拼回 + 索引
    new_table = torch.cat([new_sub, old_table[old_num - 3:]])

    key = f"{_to_int(window_size[1])},{_to_int(window_size[0])}"
    if key not in self.relative_position_indices:
        self.relative_position_indices[key] = gen_relative_position_index(window_size)

    rel_bias = new_table[self.relative_position_indices[key].view(-1)].view(
        window_size[0] * window_size[1] + 1,
        window_size[0] * window_size[1] + 1,
        -1
    )  # (Wh*Ww+1, Wh*Ww+1, nH)

    rel_bias = rel_bias.permute(2, 0, 1).contiguous()  # (nH, Wh*Ww+1, Wh*Ww+1)
    return rel_bias.unsqueeze(0)  # (1, nH, Wh*Ww+1, Wh*Ww+1)

# def _get_rel_pos_bias(self, window_size):
#     """
#     Modification of timm.models.beit.py: Attention._get_rel_pos_bias to support arbitrary window sizes.
#     """
#     old_height = 2 * self.window_size[0] - 1
#     old_width = 2 * self.window_size[1] - 1

#     new_height = 2 * window_size[0] - 1
#     new_width = 2 * window_size[1] - 1

#     old_relative_position_bias_table = self.relative_position_bias_table

#     old_num_relative_distance = self.num_relative_distance
#     new_num_relative_distance = new_height * new_width + 3

#     old_sub_table = old_relative_position_bias_table[:old_num_relative_distance - 3]

#     old_sub_table = old_sub_table.reshape(1, old_width, old_height, -1).permute(0, 3, 1, 2)
#     new_sub_table = F.interpolate(old_sub_table, size=(new_height, new_width), mode="bilinear")
#     new_sub_table = new_sub_table.permute(0, 2, 3, 1).reshape(new_num_relative_distance - 3, -1)

#     new_relative_position_bias_table = torch.cat(
#         [new_sub_table, old_relative_position_bias_table[old_num_relative_distance - 3:]])

#     key = str(window_size[1]) + "," + str(window_size[0])
#     if key not in self.relative_position_indices.keys():
#         self.relative_position_indices[key] = gen_relative_position_index(window_size)

#     relative_position_bias = new_relative_position_bias_table[
#         self.relative_position_indices[key].view(-1)].view(
#         window_size[0] * window_size[1] + 1,
#         window_size[0] * window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
#     relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#     return relative_position_bias.unsqueeze(0)


def attention_forward(self, x, resolution, shared_rel_pos_bias: Optional[torch.Tensor] = None):
    """
    Modification of timm.models.beit.py: Attention.forward to support arbitrary window sizes.
    """
    B, N, C = x.shape

    qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))

    if self.relative_position_bias_table is not None:
        window_size = tuple(np.array(resolution) // 16)
        attn = attn + self._get_rel_pos_bias(window_size)
    if shared_rel_pos_bias is not None:
        attn = attn + shared_rel_pos_bias

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# def block_forward(self, x, resolution, shared_rel_pos_bias: Optional[torch.Tensor] = None):
#     """
#     Modification of timm.models.beit.py: Block.forward to support arbitrary window sizes.
#     """
#     if self.gamma_1 is None:
#         x = x + self.drop_path(self.attn(self.norm1(x), resolution, shared_rel_pos_bias=shared_rel_pos_bias))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#     else:
#         x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), resolution,
#                                                         shared_rel_pos_bias=shared_rel_pos_bias))
#         x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
#     return x

# 2025-10-8 02:21:41
def block_forward(self, x, resolution, shared_rel_pos_bias=None):
    """
    兼容不同 timm 版本的 BEiT Block：
    - 有的版本用 drop_path（单一路径）
    - 有的版本用 drop_path1 / drop_path2（两条路径分别应用）
    - gamma_1 / gamma_2 也可能不存在（则视为 1.0）
    """
    # 取可用的 DropPath
    dp1 = getattr(self, "drop_path1", getattr(self, "drop_path", torch.nn.Identity()))
    dp2 = getattr(self, "drop_path2", getattr(self, "drop_path", torch.nn.Identity()))
    # gamma 兼容（可能没有这两个参数）
    gamma1 = getattr(self, "gamma_1", 1.0)
    gamma2 = getattr(self, "gamma_2", 1.0)

    # norm/attn/mlp 都来自 timm 的 BEiT 实现
    x = x + dp1(gamma1 * self.attn(self.norm1(x), resolution, shared_rel_pos_bias=shared_rel_pos_bias))
    x = x + dp2(gamma2 * self.mlp(self.norm2(x)))
    return x


def beit_forward_features(self, x):
    """
    Modification of timm.models.beit.py: Beit.forward_features to support arbitrary window sizes.
    """
    resolution = x.shape[2:]

    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    x = self.pos_drop(x)

    rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    for blk in self.blocks:
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
        else:
            x = blk(x, resolution, shared_rel_pos_bias=rel_pos_bias)
    x = self.norm(x)
    return x


def _make_beit_backbone(
        model,
        features=[96, 192, 384, 768],
        size=[384, 384],
        hooks=[0, 4, 8, 11],
        vit_features=768,
        use_readout="ignore",
        start_index=1,
        start_index_readout=1,
):
    backbone = make_backbone_default(model, features, size, hooks, vit_features, use_readout, start_index,
                                     start_index_readout)

    backbone.model.patch_embed.forward = types.MethodType(patch_embed_forward, backbone.model.patch_embed)
    backbone.model.forward_features = types.MethodType(beit_forward_features, backbone.model)

    for block in backbone.model.blocks:
        attn = block.attn
        attn._get_rel_pos_bias = types.MethodType(_get_rel_pos_bias, attn)
        attn.forward = types.MethodType(attention_forward, attn)
        attn.relative_position_indices = {}

        block.forward = types.MethodType(block_forward, block)

    return backbone


def _make_pretrained_beitl16_512(pretrained, use_readout="ignore", hooks=None):
    model = timm.create_model("beit_large_patch16_512", pretrained=pretrained)

    hooks = [5, 11, 17, 23] if hooks is None else hooks

    features = [256, 512, 1024, 1024]

    return _make_beit_backbone(
        model,
        features=features,
        size=[512, 512],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
    )


def _make_pretrained_beitl16_384(pretrained, use_readout="ignore", hooks=None):
    model = timm.create_model("beit_large_patch16_384", pretrained=pretrained)

    hooks = [5, 11, 17, 23] if hooks is None else hooks
    return _make_beit_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
    )


def _make_pretrained_beitb16_384(pretrained, use_readout="ignore", hooks=None):
    model = timm.create_model("beit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks is None else hooks
    return _make_beit_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
    )
