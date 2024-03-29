from functools import partial

import numpy as np
import torch
from timm.models import skresnext50_32x4d
from timm.models.dpn import dpn92, dpn131
from timm.models.efficientnet import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, tf_efficientnet_b6_ns,tf_efficientnet_b7_ns,tf_efficientnet_l2_ns_475, tf_efficientnet_l2_ns,tf_efficientnetv2_l_in21k, tf_efficientnetv2_l_in21ft1k, tf_efficientnetv2_m_in21k, tf_efficientnetv2_m_in21ft1k
from timm.models.resnest import resnest200e, resnest269e
from timm.models.vision_transformer import deit_base_distilled_patch16_384, deit_base_patch16_384
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
#from geffnet import tf_efficientnet_b7_ns
encoder_params = {
    "dpn92": {
        "features": 2688,
        "init_op": partial(dpn92, pretrained=True)
    },
    "dpn131": {
        "features": 2688,
        "init_op": partial(dpn131, pretrained=True)
    },
    "tf_efficientnet_b3_ns": {
        "features": 1536,
        "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b2_ns": {
        "features": 1408,
        "init_op": partial(tf_efficientnet_b2_ns, pretrained=False, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.5)
    },
    "tf_efficientnet_b5_ns": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns_03d": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_03d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_04d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.4)
    },
    "tf_efficientnet_b6_ns": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b6_ns_04d": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.4)
    },
    "sk50": {
        "features": 2048,
        "init_op": partial(skresnext50_32x4d, pretrained=True)
    },
    "resnest200e": {
        "features": 2048,
        "init_op": partial(resnest200e, pretrained=True)
    },
    "resnest269e": {
        "features": 2048,
        "init_op": partial(resnest269e, pretrained=True)
    },
    "tf_efficientnet_l2_ns": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_l2_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_l2_ns_475": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_l2_ns_475, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnetv2_l_in21k": {
        "features": 1280,
        "init_op": partial(tf_efficientnetv2_l_in21k, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnetv2_l_in21ft1k": {
        "features": 1280,
        "init_op": partial(tf_efficientnetv2_l_in21ft1k, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnetv2_m_in21k": {
        "features": 1280,
        "init_op": partial(tf_efficientnetv2_m_in21k, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnetv2_m_in21ft1k": {
        "features": 1280,
        "init_op": partial(tf_efficientnetv2_m_in21ft1k, pretrained=True, drop_path_rate=0.2)
    },
    "deit_base_distilled_patch16_384": {
        "features": 768,
        "init_op": partial(deit_base_distilled_patch16_384, pretrained=True, drop_path_rate=0.2)
    }, 
    "deit_base_patch16_384": {
        "features": 768,
        "init_op": partial(deit_base_patch16_384, pretrained=True, drop_path_rate=0.2)
    },
}


def setup_srm_weights(input_channels: int = 3) -> torch.Tensor:
    """Creates the SRM kernels for noise analysis."""
    # note: values taken from Zhou et al., "Learning Rich Features for Image Manipulation Detection", CVPR2018
    srm_kernel = torch.from_numpy(np.array([
        [  # srm 1/2 horiz
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 1., -2., 1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/4
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 2., -4., 2., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/12
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-2., 8., -12., 8., -2.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
        ]
    ])).float()
    srm_kernel[0] /= 2
    srm_kernel[1] /= 4
    srm_kernel[2] /= 12
    return srm_kernel.view(3, 1, 5, 5).repeat(1, input_channels, 1, 1)


def setup_srm_layer(input_channels: int = 3) -> torch.nn.Module:
    """Creates a SRM convolution layer for noise analysis."""
    weights = setup_srm_weights(input_channels)
    conv = torch.nn.Conv2d(input_channels, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(weights, requires_grad=False)
    return conv


class DeepFakeClassifierSRM(nn.Module):
    def __init__(self, encoder, dropout_rate=0.5) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.srm_conv = setup_srm_layer(3)
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        noise = self.srm_conv(x)
        x = self.encoder.forward_features(noise)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class GlobalWeightedAvgPool2d(nn.Module):
    """
    Global Weighted Average Pooling from paper "Global Weighted Average
    Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features: int, flatten=False):
        super().__init__()
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.flatten = flatten

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3], keepdim=not self.flatten)
        return x


class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        # for train
        f = open("/dataset/workspace/backup/deit_shape2.txt","a+")
        f.write(str(x.shape))
        f.write("forward begin\n")
        x = self.encoder.forward_features(x)
        # for ONNX Export
        #x = self.encoder.features(x)
        f.write("before_avg_pool :" + str(x.shape))
        x = self.avg_pool(x).flatten(1)
        f.write("after_avg_pool :" + str(x.shape))
        x = self.dropout(x)
        x = self.fc(x)
        return x

class DeepFakeClassifierWithViT(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        # for train
        f = open("/dataset/workspace/backup/new_deit_shape2.txt","a+")
        f.write(str(x.shape))
        f.write("forward begin\n")
        x = self.encoder.forward_features(x)
        f.write("after forward\n")
        f.write(str(len(x)))
        # for ONNX Export
        #x = self.encoder.features(x)
        #f.write("before fc :" + str(x.shape))
        #x = self.avg_pool(x).flatten(1)
        #f.write("after_avg_pool :" + str(x.shape))
        #x = self.dropout(x)
        x = self.fc(x)
        return x

class DeepFakeClassifierWithViTDis(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)
        self.fc_dist = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        # for train
        f = open("/dataset/workspace/backup/new_deit_shape2.txt","a+")
        x = self.encoder.forward_features(x)
        if self.encoder.head_dist is not None:
            x = list(x)
            x[0] = self.dropout(x[0])
            x[1] = self.dropout(x[1])
            x = tuple(x)
            x, x_dist = self.fc(x[0]), self.fc_dist(x[1])  # x must be a tuple
            if self.encoder.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.fc(x)
        return x
        # for ONNX Export
        #x = self.encoder.features(x)
        #f.write("before fc :" + str(x.shape))
        #x = self.avg_pool(x).flatten(1)
        #f.write("after_avg_pool :" + str(x.shape))
        #x = self.dropout(x)
        
class DeepFakeClassifierGWAP(nn.Module):
    def __init__(self, encoder, dropout_rate=0.5) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = GlobalWeightedAvgPool2d(encoder_params[encoder]["features"])
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
