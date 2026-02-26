# ============================================================
# 주제 9. 전이학습
# ============================================================
# 이 파일은 교재의 실습 코드를 추출한 것입니다.
# Colab 환경에서 실행하세요: https://colab.research.google.com


# ------------------------------------------------------------
# 01. 전이학습 개념
# ------------------------------------------------------------
from torchvision import models

model = models.resnet18(weights='IMAGENET1K_V1')

# ─── 전략 1: 특성 추출 ────────────────────────────
# 모든 레이어 동결 (학습 안 됨)
for param in model.parameters():
    param.requires_grad = False

# 마지막 분류층만 학습 가능하게 교체
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 클래스 예시

# ─── 전략 2: 전체 파인튜닝 ──────────────────────
# 다시 모든 레이어 학습 가능하게
for param in model.parameters():
    param.requires_grad = True

# ─── 전략 3: 부분 파인튜닝 (권장) ──────────────
# 초기 레이어 동결, 후반부만 학습
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 학습 가능한 파라미터 수 확인
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"학습 가능: {trainable:,} / 전체: {total:,} ({trainable/total*100:.1f}%)")

import torch.optim as optim

# 레이어별 다른 학습률 설정
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # 후반부: 작은 lr
    {'params': model.fc.parameters(),     'lr': 1e-3},  # 새 레이어: 큰 lr
])


# ------------------------------------------------------------
# 02. ResNet 파인튜닝
# ------------------------------------------------------------
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# ImageNet pretrained 모델용 정규화 (ImageNet 통계 사용)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Kaggle flowers 데이터셋 (ImageFolder 구조 가정)
full_dataset = datasets.ImageFolder(root='data/flowers', transform=train_transform)
CLASSES = full_dataset.classes
print(f"클래스: {CLASSES}")  # ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

# val_set은 train_set과 같은 dataset 객체를 공유하므로 직접 transform을 변경하면 안됩니다.
# 대신 val 전용 dataset을 별도로 생성합니다.
val_dataset = datasets.ImageFolder(root='data/flowers', transform=test_transform)
val_set = torch.utils.data.Subset(val_dataset, val_set.indices)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=2)

import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 사전학습 ResNet18 로드
model = models.resnet18(weights='IMAGENET1K_V1')

# 특성 추출기 부분: 후반 레이어(layer4)만 파인튜닝
for name, param in model.named_parameters():
    param.requires_grad = 'layer4' in name  # layer4만 학습

# 분류기 교체: ImageNet 1000 → 꽃 5종
num_features = model.fc.in_features  # 512
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 5)
)
model = model.to(device)

# 레이어별 다른 학습률
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3}
])
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

EPOCHS = 15
best_acc = 0

for epoch in range(1, EPOCHS + 1):
    # 학습
    model.train()
    train_correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_correct += outputs.max(1)[1].eq(labels).sum().item()
    scheduler.step()

    # 평가
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            val_correct += model(images).max(1)[1].eq(labels).sum().item()

    val_acc = val_correct / len(val_set) * 100
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'resnet18_flowers.pth')

    if epoch % 5 == 0:
        train_acc = train_correct / len(train_set) * 100
        print(f"Epoch {epoch:2d} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")

print(f"\n최고 검증 정확도: {best_acc:.2f}%")


# ------------------------------------------------------------
# 03. EfficientNet 파인튜닝
# ------------------------------------------------------------
import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EfficientNet-B0 (가장 작고 빠른 버전)
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
print(model.classifier)  # Sequential: Dropout → Linear(1280, 1000)

# 파라미터 수 비교
resnet18 = models.resnet18(weights=None)
print(f"ResNet18: {sum(p.numel() for p in resnet18.parameters()):,}")
print(f"EfficientNet-B0: {sum(p.numel() for p in model.parameters()):,}")

# EfficientNet-B0의 마지막 분류기 교체
# in_features = 1280 (EfficientNet-B0 기준)
num_classes = 5  # 꽃 5종

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(1280, num_classes)
)
model = model.to(device)

# 전체 파인튜닝 (데이터가 충분한 경우)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 5))
model.load_state_dict(torch.load('efficientnet_flowers.pth', map_location=device, weights_only=True))
model.eval().to(device)

CLASSES = ['데이지', '민들레', '장미', '해바라기', '튤립']

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(5)}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="꽃 사진"),
    outputs=gr.Label(num_top_classes=5, label="분류 결과"),
    title="🌸 꽃 종류 분류기 (EfficientNet 전이학습)",
    description="꽃 사진을 업로드하면 5종류 중 하나로 분류합니다."
).launch(share=True)
