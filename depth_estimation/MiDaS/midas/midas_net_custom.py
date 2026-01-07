"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import FeatureFusionBlock, FeatureFusionBlock_custom, Interpolate, _make_encoder


class MidasNet_small(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=64, backbone="efficientnet_lite3", non_negative=True, exportable=True, channels_last=False, align_corners=True,
        blocks={'expand': True}):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet_small, self).__init__()

        use_pretrained = False if path else True
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
  
        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)

        
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
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
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)


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

class MidasNetSmallWithFeatures(MidasNet_small):
    def __init__(self, *args, check_size=32, **kwargs):
        """
        参数：
          check_size: 输入高宽补齐到该数的倍数（默认32，对应下采样步长对齐）
        其它参数同 MidasNet_small
        """
        super().__init__(*args, **kwargs)
        self.check_size = int(check_size)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        s = self.check_size
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (h, w, pad_h, pad_w)

    def forward(self, x):
        """
        返回与 DPTDepthModelWithFeatures 对齐的字典：
          final_output, layer1..layer4, layer1_rn..layer4_rn, path_1..path_4
        """
        # 可选：channels_last
        if getattr(self, "channels_last", False):
            x = x.contiguous(memory_format=torch.channels_last)

        # 1) 尺寸补齐
        x, (h, w, pad_h, pad_w) = self.check_image_size(x)

        # 2) 提取主干各层特征（efficientnet_lite3 等）
        layer_1 = self.pretrained.layer1(x)          # H/4
        layer_2 = self.pretrained.layer2(layer_1)    # H/8
        layer_3 = self.pretrained.layer3(layer_2)    # H/16
        layer_4 = self.pretrained.layer4(layer_3)    # H/32

        # 3) 统一到 scratch 的 features 维度
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 4) RefineNet 自顶向下融合
        path_4 = self.scratch.refinenet4(layer_4_rn)                 # ~H/16
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)         # ~H/8
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)         # ~H/4
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)         # ~H/2

        # 5) 最终深度预测（保持 B×1×H'×W'）
        final_output = self.scratch.output_conv(path_1)

        # 6) 如需裁回原尺寸可在此处执行（保持与 DPT 字典版一致可不裁）
        # if pad_h or pad_w:
        #     final_output = final_output[..., :h, :w]

        return {
            "final_output": final_output,  # (B, 1, H', W')
            "layer1": layer_1,             # encoder 第1阶段原始特征 (B, C1, H/4,  W/4)
            "layer2": layer_2,             # (B, C2, H/8,  W/8)
            "layer3": layer_3,             # (B, C3, H/16, W/16)
            "layer4": layer_4,             # (B, C4, H/32, W/32)
            "layer1_rn": layer_1_rn,       # 统一到 features 通道后的特征
            "layer2_rn": layer_2_rn,
            "layer3_rn": layer_3_rn,
            "layer4_rn": layer_4_rn,
            "path_4": path_4,              # 自顶向下融合特征
            "path_3": path_3,
            "path_2": path_2,
            "path_1": path_1,
        }



def fuse_model(m):
    prev_previous_type = nn.Identity()
    prev_previous_name = ''
    previous_type = nn.Identity()
    previous_name = ''
    for name, module in m.named_modules():
        if prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d and type(module) == nn.ReLU:
            # print("FUSED ", prev_previous_name, previous_name, name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name, name], inplace=True)
        elif prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d:
            # print("FUSED ", prev_previous_name, previous_name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name], inplace=True)
        # elif previous_type == nn.Conv2d and type(module) == nn.ReLU:
        #    print("FUSED ", previous_name, name)
        #    torch.quantization.fuse_modules(m, [previous_name, name], inplace=True)

        prev_previous_type = previous_type
        prev_previous_name = previous_name
        previous_type = type(module)
        previous_name = name