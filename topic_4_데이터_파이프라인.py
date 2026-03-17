# ============================================================
# 주제 4. 데이터 파이프라인
# ============================================================
# 이 파일은 교재의 실습 코드를 추출한 것입니다.
# Colab 환경에서 실행하세요: https://colab.research.google.com


# ------------------------------------------------------------
# 01. DataLoader 이해하기
# ------------------------------------------------------------
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=128,     # 배치 크기: GPU 메모리에 맞게 설정
    shuffle=True,       # 학습 시 True, 평가 시 False
    num_workers=4,      # 데이터 로딩 병렬 프로세스 수
    pin_memory=True,    # GPU 사용 시 True (CPU→GPU 전송 가속)
    drop_last=False,    # 마지막 불완전 배치 버릴지 여부
)

import time
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)

for bs in [32, 64, 128, 256]:
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=2)
    start = time.time()
    for images, _ in loader:
        pass  # 실제 학습 없이 로딩 속도만 측정
    elapsed = time.time() - start
    print(f"batch_size={bs:3d}: {elapsed:.2f}초, {len(loader)}배치")

from torch.utils.data import WeightedRandomSampler
import numpy as np

# 클래스별 샘플 수가 다를 때 (예: 정상 900개, 불량 100개)
labels = [0] * 900 + [1] * 100  # 예시 레이블

# 클래스별 가중치 계산 (적은 클래스에 높은 가중치)
class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

loader = DataLoader(dataset, batch_size=64, sampler=sampler)
# sampler 사용 시 shuffle 옵션 불필요

from torch.utils.data import random_split

# 학습 데이터를 8:2로 학습/검증 분리
dataset = datasets.CIFAR10('./data', train=True, download=True,
                           transform=transform)

train_size = int(0.8 * len(dataset))  # 40,000
val_size   = len(dataset) - train_size  # 10,000

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False, num_workers=2)

print(f"학습: {len(train_set)}장 / 검증: {len(val_set)}장")


# ------------------------------------------------------------
# 02. 손실함수와 옵티마이저
# ------------------------------------------------------------
import torch
import torch.nn as nn

# 다중 분류 (클래스 10개, 배치 4)
criterion_ce = nn.CrossEntropyLoss()
logits = torch.rand(4, 10)     # 모델 출력 (softmax 전)
labels = torch.randint(0, 10, (4,))
loss = criterion_ce(logits, labels)
print(f"CrossEntropyLoss: {loss.item():.4f}")

# 이진 분류
criterion_bce = nn.BCEWithLogitsLoss()
logits_bin = torch.rand(4, 1)   # 모델 출력 (sigmoid 전)
labels_bin = torch.randint(0, 2, (4, 1)).float()
loss_bin = criterion_bce(logits_bin, labels_bin)
print(f"BCEWithLogitsLoss: {loss_bin.item():.4f}")

import torch.optim as optim

model = SimpleNet()  # 앞서 정의한 모델

# SGD (Stochastic Gradient Descent)
optimizer_sgd = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,      # 이전 기울기 방향 유지
    weight_decay=1e-4  # L2 정규화 (과적합 방지)
)

# Adam (Adaptive Moment Estimation)
optimizer_adam = optim.Adam(
    model.parameters(),
    lr=0.001,          # SGD보다 작은 학습률 사용
    betas=(0.9, 0.999),# 기울기와 제곱 기울기의 지수이동평균 계수
    weight_decay=1e-4
)

# Colab 한글 폰트 설정
import subprocess
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

subprocess.run(['apt-get', '-qq', 'install', '-y', 'fonts-nanum'],
               capture_output=True, check=True)

fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

# 학습률별 손실 변화 시뮬레이션
results = {}
for lr in [0.1, 0.01, 0.001, 0.0001]:
    model_temp = SimpleNet()
    optimizer_temp = optim.Adam(model_temp.parameters(), lr=lr)
    losses = []

    for _ in range(100):
        x = torch.rand(64, 784)
        y = torch.randint(0, 10, (64,))
        out = model_temp(x.view(64, 1, 28, 28))
        loss = nn.CrossEntropyLoss()(out, y)
        optimizer_temp.zero_grad()
        loss.backward()
        optimizer_temp.step()
        losses.append(loss.item())
    results[lr] = losses

plt.figure(figsize=(10, 5))
for lr, losses in results.items():
    plt.plot(losses, label=f'lr={lr}')
plt.xlabel('스텝')
plt.ylabel('손실')
plt.title('학습률별 손실 변화')
plt.legend()
plt.show()


# ------------------------------------------------------------
# 03. Dropout으로 과적합 방지
# ------------------------------------------------------------
import matplotlib.pyplot as plt

# 학습 기록
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# (학습 후 기록된 값이 있다고 가정)
# 학습 곡선으로 과적합 확인
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='학습 손실')
ax1.plot(val_losses, label='검증 손실')
ax1.set_title('손실 곡선')
ax1.set_xlabel('에포크')
ax1.legend()

ax2.plot(train_accs, label='학습 정확도')
ax2.plot(val_accs, label='검증 정확도')
ax2.set_title('정확도 곡선')
ax2.set_xlabel('에포크')
ax2.legend()
plt.tight_layout()
plt.show()

import torch.nn as nn

class NetWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(p=0.5)   # 50% 뉴런 랜덤으로 비활성화
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.3)   # 30% 비활성화
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        return self.fc3(x)

# model.train() 시에만 Dropout 적용
# model.eval() 시에는 Dropout 비활성화 (자동)
model = NetWithDropout()

class CNN_with_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),    # Conv 뒤에 BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.BatchNorm1d(512),   # Linear 뒤에 BatchNorm
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience    # 몇 에포크 기다릴지
        self.min_delta = min_delta  # 개선으로 인정할 최소 변화량
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # 계속 학습
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 학습 중단
            return False

# 사용 예시
early_stopping = EarlyStopping(patience=5)
for epoch in range(100):
    # ... 학습 ...
    val_loss = ...  # 검증 손실 계산
    if early_stopping(val_loss):
        print(f"Early Stopping: {epoch}에포크에서 중단")
        break
