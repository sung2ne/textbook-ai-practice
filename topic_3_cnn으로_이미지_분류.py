# ============================================================
# 주제 3. CNN으로 이미지 분류
# ============================================================
# 이 파일은 교재의 실습 코드를 추출한 것입니다.
# Colab 환경에서 실행하세요: https://colab.research.google.com


# ------------------------------------------------------------
# 01. CNN이란 무엇인가
# ------------------------------------------------------------
import torch
import torch.nn as nn

# Conv2d(in_channels, out_channels, kernel_size, padding)
conv = nn.Conv2d(
    in_channels=3,    # 입력 채널 (RGB 컬러: 3)
    out_channels=32,  # 출력 채널 (필터 32개)
    kernel_size=3,    # 3×3 필터
    padding=1         # 출력 크기 유지
)

# 32×32 컬러 이미지 1장
x = torch.rand(1, 3, 32, 32)
out = conv(x)
print(f"입력: {x.shape}")    # [1, 3, 32, 32]
print(f"출력: {out.shape}")  # [1, 32, 32, 32] - 채널 수만 변화

pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2×2 영역, 2칸씩 이동

x = torch.rand(1, 32, 32, 32)
out = pool(x)
print(f"입력: {x.shape}")    # [1, 32, 32, 32]
print(f"출력: {out.shape}")  # [1, 32, 16, 16] - 크기 절반으로 축소

# 전형적인 CNN 블록
block = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),   # 3채널 → 32채널
    nn.ReLU(),
    nn.MaxPool2d(2, 2),               # 32×32 → 16×16
)

x = torch.rand(1, 3, 32, 32)
out = block(x)
print(f"블록 통과 후: {out.shape}")  # [1, 32, 16, 16]


# ------------------------------------------------------------
# 02. CIFAR-10 분류 구현
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10 전처리 (학습용: 증강 포함)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),           # 좌우 반전 (데이터 증강)
    transforms.RandomCrop(32, padding=4),        # 랜덤 크롭
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),   # RGB 평균
                         (0.2023, 0.1994, 0.2010))   # RGB 표준편차
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10('./data', train=True,  download=True, transform=train_transform)
test_dataset  = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=2)

classes = ['비행기', '자동차', '새', '고양이', '사슴',
           '개', '개구리', '말', '배', '트럭']

print(f"학습 데이터: {len(train_dataset)}장")
print(f"테스트 데이터: {len(test_dataset)}장")

class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 합성곱 블록 1: 3채널 → 32채널
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 32×32 → 16×16
        )
        # 합성곱 블록 2: 32채널 → 64채널
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 16×16 → 8×8
        )
        # 합성곱 블록 3: 64채널 → 128채널
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 8×8 → 4×4
        )
        # 완전연결층
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR_CNN().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"파라미터 수: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    # 학습
    model.train()
    train_loss, train_correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += outputs.max(1)[1].eq(labels).sum().item()

    # 평가
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += outputs.max(1)[1].eq(labels).sum().item()

    if epoch % 5 == 0:
        train_acc = train_correct / len(train_dataset) * 100
        test_acc  = test_correct  / len(test_dataset)  * 100
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")


# ------------------------------------------------------------
# 03. 필터 시각화
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# 학습된 모델의 첫 번째 Conv 레이어 가중치
weights = model.block1[0].weight.data.cpu()  # [32, 3, 3, 3]
print(f"필터 shape: {weights.shape}")  # [필터수, 채널, H, W]

# 필터 정규화 (시각화용)
w_min, w_max = weights.min(), weights.max()
weights = (weights - w_min) / (w_max - w_min)

# 32개 필터 시각화
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < weights.shape[0]:
        # RGB 채널 합성 (3채널 → 시각화)
        filt = weights[i].permute(1, 2, 0).numpy()  # [3,3,3] → [3,3,3] HWC
        ax.imshow(filt)
    ax.axis('off')
plt.suptitle("첫 번째 합성곱 레이어 필터 (32개)", fontsize=12)
plt.tight_layout()
plt.show()

# 테스트 이미지 1장 가져오기
image, label = test_dataset[0]
image_tensor = image.unsqueeze(0).to(device)  # [1, 3, 32, 32]

# 첫 번째 블록의 출력 추출 (hook 사용)
feature_maps = {}

def hook_fn(module, input, output):
    feature_maps['block1'] = output.detach().cpu()

hook = model.block1.register_forward_hook(hook_fn)

# 순전파 실행
model.eval()
with torch.no_grad():
    _ = model(image_tensor)

hook.remove()

# 특징 맵 시각화 (32개 채널 중 16개)
fmaps = feature_maps['block1'][0]  # [32, 16, 16]
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(fmaps[i].numpy(), cmap='viridis')
    ax.set_title(f"필터 {i+1}", fontsize=8)
    ax.axis('off')

# 원본 이미지도 함께 표시
plt.suptitle(f"원본: {classes[label]} | 첫 번째 블록 특징 맵", fontsize=12)
plt.tight_layout()
plt.show()

# 각 클래스별로 올바르게 예측한 예시와 틀린 예시 시각화
model.eval()
class_correct = {c: [] for c in classes}
class_wrong   = {c: [] for c in classes}

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        preds = outputs.max(1)[1].cpu()
        for img, true, pred in zip(images, labels, preds):
            cls = classes[true.item()]
            if true == pred and len(class_correct[cls]) < 2:
                class_correct[cls].append((img, pred.item()))
            elif true != pred and len(class_wrong[cls]) < 2:
                class_wrong[cls].append((img, pred.item()))

# 클래스별 맞춘 사례 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, (cls, ax) in enumerate(zip(classes, axes.flat)):
    if class_correct[cls]:
        img, pred = class_correct[cls][0]
        ax.imshow(img.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        ax.set_title(f"{cls}\n예측: {classes[pred]}", fontsize=8)
    ax.axis('off')
plt.suptitle("클래스별 정답 예시", fontsize=12)
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        preds = outputs.max(1)[1].cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('예측')
plt.ylabel('정답')
plt.title('CIFAR-10 혼동 행렬')
plt.tight_layout()
plt.show()
