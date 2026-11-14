"""Model definition: EfficientNet-B0 backbone with a replaced classification head."""
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(num_classes: int, pretrained: bool = True,
                freeze_backbone: bool = True, unfreeze_last_n: int = 2,
                dropout: float = 0.3) -> nn.Module:
    """
    Args:
        freeze_backbone: freeze the convolutional feature extractor.
        unfreeze_last_n: keep the last N feature blocks trainable (they hold the
            most task-specific, high-level features). 0 = pure linear probe.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            for block in list(model.features)[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True

    in_features = model.classifier[1].in_features          # 1280 for B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),               # new head (trainable)
    )
    return model


def param_groups(model: nn.Module, head_lr: float, backbone_lr: float):
    """Higher LR for the fresh head, lower LR for the unfrozen backbone blocks."""
    head = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    groups = [{"params": head, "lr": head_lr}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    return groups


# --- ResNet-50 alternative -------------------------------------------------
# from torchvision.models import resnet50, ResNet50_Weights
# m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# for n, p in m.named_parameters():
#     p.requires_grad = n.startswith(("layer4", "fc"))
# m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
# (then use m.fc / m.layer4 in param_groups instead of classifier / features)
