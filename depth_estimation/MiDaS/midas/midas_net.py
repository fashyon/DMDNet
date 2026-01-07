"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import FeatureFusionBlock, Interpolate, _make_encoder


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(backbone="resnext101_wsl", features=features, use_pretrained=use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)

class MidasNetWithFeatures(MidasNet):
    def __init__(self, *args, check_size, **kwargs):
        """
        参数：
          check_size: 输入高宽补齐到该数的倍数（默认32，对应下采样步长对齐）
        其它参数同 MidasNet
        """
        super().__init__(*args, **kwargs)
        self.check_size = check_size

    def check_image_size(self, x):
        _, _, h, w = x.size()
        s = int(self.check_size)
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (h, w, pad_h, pad_w)

    def forward(self, x):
        # 1) 尺寸检查 & 补齐
        x, (h, w, pad_h, pad_w) = self.check_image_size(x)

        # 2) 提取主干各层特征（ResNeXt101-WSL）
        layer_1 = self.pretrained.layer1(x)  # (B, 256,  H/4,  W/4)
        layer_2 = self.pretrained.layer2(layer_1)  # (B, 512,  H/8,  W/8)
        layer_3 = self.pretrained.layer3(layer_2)  # (B, 1024, H/16, W/16)
        layer_4 = self.pretrained.layer4(layer_3)  # (B, 2048, H/32, W/32)

        # 3) 投影到统一通道维度（features，默认256）
        layer_1_rn = self.scratch.layer1_rn(layer_1)  # (B, features, H/4,  W/4)
        layer_2_rn = self.scratch.layer2_rn(layer_2)  # (B, features, H/8,  W/8)
        layer_3_rn = self.scratch.layer3_rn(layer_3)  # (B, features, H/16, W/16)
        layer_4_rn = self.scratch.layer4_rn(layer_4)  # (B, features, H/32, W/32)

        # 4) Refinenet 自顶向下融合
        path_4 = self.scratch.refinenet4(layer_4_rn)  # (B, features, H/16, W/16)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)  # (B, features, H/8,  W/8)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)  # (B, features, H/4,  W/4)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)  # (B, features, H/2,  W/2)

        # 5) 头部得到深度图（不 squeeze，保持 B×1×H'×W'）
        final_output = self.scratch.output_conv(path_1)  # (B, 1, H, W) after upsample

        # 6) 如有补边，这里可选裁回原尺寸（保持与 DPT 版本一致可不裁）
        # if pad_h or pad_w:
        #     final_output = final_output[..., :h, :w]

        return {
            "final_output": final_output,  # (B, 1, H', W')
            "layer1": layer_1,  # (B, 256,  H/4,  W/4)
            "layer2": layer_2,  # (B, 512,  H/8,  W/8)
            "layer3": layer_3,  # (B, 1024, H/16, W/16)
            "layer4": layer_4,  # (B, 2048, H/32, W/32)
            "layer1_rn": layer_1_rn,  # (B, features, H/4,  W/4)
            "layer2_rn": layer_2_rn,  # (B, features, H/8,  W/8)
            "layer3_rn": layer_3_rn,  # (B, features, H/16, W/16)
            "layer4_rn": layer_4_rn,  # (B, features, H/32, W/32)
            "path_4": path_4,  # (B, features, H/16, W/16)
            "path_3": path_3,  # (B, features, H/8,  W/8)
            "path_2": path_2,  # (B, features, H/4,  W/4)
            "path_1": path_1,  # (B, features, H/2,  W/2)
        }

