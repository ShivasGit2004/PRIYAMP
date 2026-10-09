import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CNNBaseline(nn.Module):
    """Standard ResNet-18 baseline model."""
    def __init__(self, num_classes=2, dropout_rate=0.4, pretrained=True):
        super(CNNBaseline, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, return_features=False):
        feats = self.backbone(x)
        dropped = self.dropout(feats)
        logits = self.fc(dropped)
        if return_features:
            return logits, feats
        return logits


class MultiScaleDomainPCOSNet(nn.Module):
    """
    Multi-Scale Feature Fusion + SE Channel Attention + Deep CORAL Domain Adaptation
    with Monte Carlo Dropout for uncertainty estimation.
    """
    def __init__(self, num_classes=2, dropout_rate=0.4, pretrained=True, use_se=True, use_multiscale=True):
        super(MultiScaleDomainPCOSNet, self).__init__()
        self.use_se = use_se
        self.use_multiscale = use_multiscale
        
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        
        # Stem
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        # Stages
        self.layer1 = base.layer1  # 64 channels, 56x56
        self.layer2 = base.layer2  # 128 channels, 28x28
        self.layer3 = base.layer3  # 256 channels, 14x14
        self.layer4 = base.layer4  # 512 channels, 7x7
        
        # SE Attention blocks
        if self.use_se:
            self.se1 = SEBlock(64)
            self.se2 = SEBlock(128)
            self.se3 = SEBlock(256)
            self.se4 = SEBlock(512)
        
        # Multi-scale pooling and projection
        if self.use_multiscale:
            self.pool_scale2 = nn.AdaptiveAvgPool2d(1)
            self.proj_scale2 = nn.Linear(128, 128)
            
            self.pool_scale3 = nn.AdaptiveAvgPool2d(1)
            self.proj_scale3 = nn.Linear(256, 128)
            
            self.pool_scale4 = nn.AdaptiveAvgPool2d(1)
            self.proj_scale4 = nn.Linear(512, 256)
            
            total_feature_dim = 128 + 128 + 256  # 512
        else:
            self.pool_scale4 = nn.AdaptiveAvgPool2d(1)
            total_feature_dim = 512

        # Domain adaptation bottleneck embedding layer
        self.bottleneck = nn.Sequential(
            nn.Linear(total_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        # Classifier with Monte Carlo Dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(256, num_classes)
        
        # Gradient hook placeholders for Grad-CAM
        self.gradient = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradient = grad

    def forward(self, x, return_features=False):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Stage 1
        x1 = self.layer1(x)
        if self.use_se:
            x1 = self.se1(x1)
            
        # Stage 2
        x2 = self.layer2(x1)
        if self.use_se:
            x2 = self.se2(x2)
            
        # Stage 3
        x3 = self.layer3(x2)
        if self.use_se:
            x3 = self.se3(x3)
            
        # Stage 4
        x4 = self.layer4(x3)
        if self.use_se:
            x4 = self.se4(x4)
            
        # Hook for Grad-CAM
        if x4.requires_grad:
            h = x4.register_hook(self.activations_hook)
        self.activations = x4
        
        # Multi-scale feature aggregation
        if self.use_multiscale:
            f2 = self.proj_scale2(self.pool_scale2(x2).view(x2.size(0), -1))
            f3 = self.proj_scale3(self.pool_scale3(x3).view(x3.size(0), -1))
            f4 = self.proj_scale4(self.pool_scale4(x4).view(x4.size(0), -1))
            fused = torch.cat([f2, f3, f4], dim=1)
        else:
            fused = self.pool_scale4(x4).view(x4.size(0), -1)
            
        # Bottleneck domain feature
        domain_feats = self.bottleneck(fused)
        
        # Classification
        dropped = self.dropout(domain_feats)
        logits = self.fc(dropped)
        
        if return_features:
            return logits, domain_feats
        return logits


def compute_covariance(features):
    """
    Compute sample covariance matrix for Deep CORAL.
    features: (N, D)
    """
    n = features.size(0)
    if n <= 1:
        return torch.zeros((features.size(1), features.size(1)), device=features.device)
    
    # Center the features
    ones = torch.ones(n, 1, device=features.device)
    mean = torch.mm(ones.t(), features) / n
    centered = features - mean.expand_as(features)
    
    # Covariance matrix: C = (centered^T * centered) / (n - 1)
    cov = torch.mm(centered.t(), centered) / (n - 1)
    return cov


def coral_loss(source_features, target_features):
    """
    Deep CORAL correlation alignment loss between source and target representations.
    L_CORAL = 1 / (4 * d^2) * || C_S - C_T ||_F^2
    """
    d = source_features.size(1)
    cov_source = compute_covariance(source_features)
    cov_target = compute_covariance(target_features)
    
    loss = torch.norm(cov_source - cov_target, p='fro') ** 2
    loss = loss / (4.0 * (d ** 2))
    return loss
