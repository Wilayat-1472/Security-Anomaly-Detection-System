import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import torchvision.models as models


class ConvNeXtSecurityDetector(nn.Module):
    """
    State-of-the-art vision backbone (ConvNeXt-Tiny) adapted for surveillance security threat classification.
    ConvNeXt incorporates modern vision transformer architectural design choices while retaining standard conv efficiency.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[2].in_features
        # Layer norm -> Dropout -> Linear -> GELU -> Dropout -> Linear
        self.backbone.classifier = nn.Sequential(
            models.convnext.LayerNorm2d((in_features,), eps=1e-6),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        self._name = "convnext_tiny_security"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer_for_cam(self):
        # Last convolutional block in features
        return self.backbone.features[-1]


class SecurityAnomalyDetector(nn.Module):
    """
    MobileNetV2-based lightweight architecture designed for edge deployment and fast real-time CCTV analysis.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        self._name = "mobilenetv2_security"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer_for_cam(self):
        # Last convolutional feature layer before pooling
        return self.backbone.features[-1]

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNetSecurityDetector(nn.Module):
    """
    ResNet-50 backbone providing a standard research benchmark for comparative ablation studies.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        self._name = "resnet50_security"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer_for_cam(self):
        return self.backbone.layer4[-1]


def create_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    arch = model_cfg.get("name", "mobilenetv2").lower()
    num_classes = model_cfg.get("num_classes", 5)
    pretrained = model_cfg.get("pretrained", True)
    freeze_backbone = model_cfg.get("freeze_backbone", False)
    dropout = model_cfg.get("dropout", 0.3)

    if arch in ["mobilenetv2", "mobilenet_v2"]:
        return SecurityAnomalyDetector(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
    elif arch in ["convnext", "convnext_tiny"]:
        return ConvNeXtSecurityDetector(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
    elif arch in ["resnet50", "resnet"]:
        return ResNetSecurityDetector(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {arch}. Choose mobilenetv2, convnext_tiny, or resnet50.")
