# ============================================================
# 주제 7. CNN 심화
# ============================================================
# 이 파일은 교재의 실습 코드를 추출한 것입니다.
# Colab 환경에서 실행하세요: https://colab.research.google.com


# ------------------------------------------------------------
# 01. VGG 구조 분석
# ------------------------------------------------------------
import torch.nn as nn

def vgg_block(in_channels, out_channels, num_convs=2):
    """VGG 기본 블록: Conv × n + MaxPool"""
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels if i == 0 else out_channels,
            out_channels, 3, padding=1
        ))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(2, 2))  # 이미지 크기 절반으로
    return nn.Sequential(*layers)


class MiniVGG(nn.Module):
    """CIFAR-10용 간소화된 VGG"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            vgg_block(3,  64,  2),   # 32×32 → 16×16
            vgg_block(64, 128, 2),   # 16×16 → 8×8
            vgg_block(128, 256, 3),  # 8×8  → 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = MiniVGG()
print(model)

# 파라미터 수 확인
total = sum(p.numel() for p in model.parameters())
print(f"\n파라미터 수: {total:,}")

from torchvision import models

# VGG16 사전학습 모델 로드
vgg16 = models.vgg16(weights='IMAGENET1K_V1')
print(vgg16)

# 구조 확인
print("\n특징 추출부:")
print(vgg16.features)
print("\n분류부:")
print(vgg16.classifier)

# 파라미터 수
total = sum(p.numel() for p in vgg16.parameters())
print(f"\n총 파라미터: {total:,}")  # 약 1억 3천만 개


# ------------------------------------------------------------
# 02. ResNet과 Skip Connection
# ------------------------------------------------------------
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """ResNet의 기본 블록 (Bottleneck 없는 기본형)"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x            # 입력을 그대로 저장 (Skip Connection)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity    # 잔차 연결: 원본 입력을 더함
        out = self.relu(out)
        return out


# 테스트
block = ResidualBlock(64)
x = torch.rand(1, 64, 32, 32)
out = block(x)
print(f"입력: {x.shape} → 출력: {out.shape}")  # 같은 shape 유지

from torchvision import models

# ResNet18 (가장 작고 빠른 버전)
resnet18 = models.resnet18(weights='IMAGENET1K_V1')
print(resnet18)

# 레이어 구조 확인
for name, module in resnet18.named_children():
    print(f"{name}: {type(module).__name__}")
