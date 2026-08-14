import torch
import torch.nn as nn
import timm
import numpy as np
from scipy.spatial.transform import Rotation


class SpacecraftPoseModel(nn.Module):
    """
    EfficientNet-B3 backbone with custom neck and dual heads.
    Matches checkpoint architecture exactly.
    """

    def __init__(self, backbone_name: str = 'efficientnet_b3',
                 pretrained: bool = False):
        super().__init__()

        # Backbone — EfficientNet B3
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=False,
            num_classes=0  # remove classifier
        )

        # Get backbone output size
        # EfficientNet B3 outputs 1536 features
        backbone_out = 1536

        # Neck — matches checkpoint: neck.0 (1536->512), neck.3 (512->256)
        self.neck = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.SiLU(),
        )

        # Output heads
        self.quat_head  = nn.Linear(256, 4)   # quaternion [w,x,y,z]
        self.trans_head = nn.Linear(256, 3)   # translation [x,y,z]

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        quat  = self.quat_head(features)
        trans = self.trans_head(features)
        # Normalize quaternion to unit length
        quat = quat / (quat.norm(dim=1, keepdim=True) + 1e-8)
        return quat, trans


class PoolAndFlatten(nn.Module):
    """
    Helper layer to perform adaptive average pooling and flatten the output.
    """
    def forward(self, x):
        return torch.flatten(nn.functional.adaptive_avg_pool2d(x, (1, 1)), 1)


class PoseNet_ResNet50(nn.Module):
    """
    ResNet-50 backbone with custom neck and heads.
    Matches the 101MB checkpoint architecture exactly.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        import torchvision.models as models

        # Load ResNet-50 backbone (without average pooling and fc layers)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Custom Neck
        self.neck = nn.Sequential(
            PoolAndFlatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU()
        )

        # Dual output heads
        self.quat_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 4)
        )

        self.trans_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        quat = self.quat_head(features)
        trans = self.trans_head(features)
        # Normalize quaternion to unit length
        quat = quat / (quat.norm(dim=1, keepdim=True) + 1e-8)
        return quat, trans