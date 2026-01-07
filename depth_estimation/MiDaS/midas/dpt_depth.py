import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_beit,
    forward_swin,
    forward_levit,
    forward_vit,
)
from .backbones.levit import stem_b4_transpose
from timm.models.layers import get_act_layer


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        # For the Swin, Swin 2, LeViT and Next-ViT Transformers, the hierarchical architectures prevent setting the 
        # hooks freely. Instead, the hooks have to be chosen according to the ranges specified in the comments.
        hooks = {
            "beitl16_512": [5, 11, 17, 23],
            "beitl16_384": [5, 11, 17, 23],
            "beitb16_384": [2, 5, 8, 11],
            "swin2l24_384": [1, 1, 17, 1],  # Allowed ranges: [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "swin2b24_384": [1, 1, 17, 1],                  # [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "swin2t16_256": [1, 1, 5, 1],                   # [0, 1], [0,  1], [ 0,  5], [ 0,  1]
            "swinl12_384": [1, 1, 17, 1],                   # [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "next_vit_large_6m": [2, 6, 36, 39],            # [0, 2], [3,  6], [ 7, 36], [37, 39]
            "levit_384": [3, 11, 21],                       # [0, 3], [6, 11], [14, 21]
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }[backbone]

        if "next_vit" in backbone:
            in_features = {
                "next_vit_large_6m": [96, 256, 512, 1024],
            }[backbone]
        else:
            in_features = None

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False, # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks,
            use_readout=readout,
            in_features=in_features,
        )

        self.number_layers = len(hooks) if hooks is not None else 4
        size_refinenet3 = None
        self.scratch.stem_transpose = None

        if "beit" in backbone:
            self.forward_transformer = forward_beit
        elif "swin" in backbone:
            self.forward_transformer = forward_swin
        elif "next_vit" in backbone:
            from .backbones.next_vit import forward_next_vit
            self.forward_transformer = forward_next_vit
        elif "levit" in backbone:
            self.forward_transformer = forward_levit
            size_refinenet3 = 7
            self.scratch.stem_transpose = stem_b4_transpose(256, 128, get_act_layer("hard_swish"))
        else:
            self.forward_transformer = forward_vit

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, size_refinenet3)
        if self.number_layers >= 4:
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head


    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)
        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
        else:
            layer_1, layer_2, layer_3, layer_4 = layers

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        if self.number_layers >= 4:
            layer_4_rn = self.scratch.layer4_rn(layer_4)

        if self.number_layers == 3:
            path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.scratch.stem_transpose is not None:
            path_1 = self.scratch.stem_transpose(path_1)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        head_features_1 = kwargs["head_features_1"] if "head_features_1" in kwargs else features
        head_features_2 = kwargs["head_features_2"] if "head_features_2" in kwargs else 32
        kwargs.pop("head_features_1", None)
        kwargs.pop("head_features_2", None)

        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
           self.load(path)

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)


class DPTDepthModelWithFeatures(DPTDepthModel):
    def __init__(self, *args, check_size, **kwargs):
        """
        参数说明：
            check_size: 控制图像尺寸补齐的单位。比如如果设置为16，
                        则图像尺寸会被填充到能被16整除。
        """
        super().__init__(*args, **kwargs)
        self.check_size = check_size

    def check_image_size(self, x):
        """
        检查输入图像尺寸，不满足 (H % check_size == 0, W % check_size == 0) 时，
        使用反射填充使其满足要求，并返回原始尺寸。
        """
        _, _, h, w = x.size()
        size = self.check_size
        mod_pad_h = (size - h % size) % size
        mod_pad_w = (size - w % size) % size
        x = torch.nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        ori_size = [h, w]
        return x, ori_size

    def forward(self, x):
        # 检查图像尺寸，并对不满足要求的图像进行填充
        x, ori_size = self.check_image_size(x)

        # 获取 transformer 输出的特征层
        layers = self.forward_transformer(self.pretrained, x)
        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
            layer_4 = None
        else:
            layer_1, layer_2, layer_3, layer_4 = layers

        # 经过各层 rn 处理
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        if self.number_layers >= 4:
            layer_4_rn = self.scratch.layer4_rn(layer_4)
        else:
            layer_4_rn = None

        # Refinenet 融合
        if self.number_layers == 3:
            path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.scratch.stem_transpose is not None:
            path_1 = self.scratch.stem_transpose(path_1)

        # 最终深度预测输出
        final_output = self.scratch.output_conv(path_1)

        # 返回包含各层特征的字典
        return {
            "final_output": final_output,  # 深度图，形状 (B, 1, H', W')
            "layer1": layer_1,            # Transformer 第一阶段输出，通道数例如 96，空间尺寸约为 (H/4, W/4)
            "layer2": layer_2,            # Transformer 第二阶段输出，通道数例如 256，空间尺寸约为 (H/8, W/8)
            "layer3": layer_3,            # Transformer 第三阶段输出，通道数例如 512，空间尺寸约为 (H/16, W/16)
            "layer4": layer_4,            # Transformer 第四阶段输出，通道数例如 1024，空间尺寸约为 (H/32, W/32)，若模型只有 3 层则为 None
            "layer1_rn": layer_1_rn,      # 经过 rn 层处理后的 layer1 特征
            "layer2_rn": layer_2_rn,      # 经过 rn 层处理后的 layer2 特征
            "layer3_rn": layer_3_rn,      # 经过 rn 层处理后的 layer3 特征
            "layer4_rn": layer_4_rn,      # 经过 rn 层处理后的 layer4 特征
            "path_3": path_3,            # Refinenet3 融合后的特征，分辨率与 layer2 接近,(B, C, H/8, W/8)
            "path_2": path_2,            # Refinenet2 融合后的特征，分辨率与 layer1 接近(B, C, H/4, W/4)
            "path_1": path_1,            # Refinenet1 融合后的高分辨率特征(B, C, H/2, W/2)
        }
