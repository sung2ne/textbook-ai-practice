# ============================================================
# 주제 2. MNIST 분류 바로 실행
# ============================================================
# 이 파일은 교재의 실습 코드를 추출한 것입니다.
# Colab 환경에서 실행하세요: https://colab.research.google.com


# ------------------------------------------------------------
# 01. 신경망의 전체 그림
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ① 데이터 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=2)

# ② 모델 설계
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)  # 28×28 → 784 flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)

# ③ 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ③ 학습 루프
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset) * 100
    return avg_loss, accuracy

# ④ 평가
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset) * 100
    return avg_loss, accuracy

EPOCHS = 5
best_acc = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')

print(f"\n최고 테스트 정확도: {best_acc:.2f}%")


# ------------------------------------------------------------
# 02. MNIST 데이터 준비
# ------------------------------------------------------------
from torchvision import datasets, transforms

# 전처리 파이프라인 정의
transform = transforms.Compose([
    transforms.ToTensor(),                        # PIL Image → 텐서 (0~1 범위)
    transforms.Normalize((0.1307,), (0.3081,))   # 정규화: (mean, std)
])

# 학습용 데이터
train_dataset = datasets.MNIST(
    root='./data',    # 저장 경로
    train=True,       # 학습용 60,000장
    download=True,    # 없으면 자동 다운로드
    transform=transform
)

# 테스트용 데이터
test_dataset = datasets.MNIST(
    root='./data',
    train=False,      # 테스트용 10,000장
    download=True,
    transform=transform
)

print(f"학습 데이터: {len(train_dataset)}장")
print(f"테스트 데이터: {len(test_dataset)}장")
print(f"이미지 shape: {train_dataset[0][0].shape}")  # [1, 28, 28]

import matplotlib.pyplot as plt

# 처음 9개 이미지 시각화
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    ax.imshow(image.squeeze(), cmap='gray')  # [1,28,28] → [28,28]
    ax.set_title(f"label: {label}", fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.show()

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=64,    # 한 번에 64장씩 처리
    shuffle=True,     # 학습 데이터는 매 에포크마다 섞기
    num_workers=2     # 데이터 로딩 병렬 처리
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,    # 테스트 데이터는 섞지 않음
    num_workers=2
)

# 배치 하나 확인
images, labels = next(iter(train_loader))
print(f"배치 이미지 shape: {images.shape}")  # [64, 1, 28, 28]
print(f"배치 레이블 shape: {labels.shape}")  # [64]
print(f"레이블 예시: {labels[:10]}")


# ------------------------------------------------------------
# 03. 신경망 설계하기
# ------------------------------------------------------------
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 레이어 정의
        self.fc1 = nn.Linear(784, 256)  # 입력 784 → 출력 256
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)   # 출력 10 (숫자 0~9)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 순전파 정의: 데이터가 레이어를 통과하는 순서
        x = x.view(-1, 784)      # [batch, 1, 28, 28] → [batch, 784]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)           # 마지막 레이어는 활성화함수 없음
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)

# 모델 구조 출력
print(model)

# 학습 가능한 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n전체 파라미터: {total_params:,}")
print(f"학습 가능 파라미터: {trainable_params:,}")

# 배치 크기 4, MNIST 이미지 shape로 더미 데이터 생성
dummy_input = torch.rand(4, 1, 28, 28).to(device)

# 순전파
output = model(dummy_input)
print(f"입력 shape: {dummy_input.shape}")   # [4, 1, 28, 28]
print(f"출력 shape: {output.shape}")         # [4, 10]
print(f"출력 예시:\n{output[:2]}")


# ------------------------------------------------------------
# 04. 학습하고 평가하기
# ------------------------------------------------------------
import torch.optim as optim

# 손실함수: CrossEntropyLoss (다중 분류에 적합)
criterion = nn.CrossEntropyLoss()

# 옵티마이저: Adam (학습률 자동 조정, 기본값 lr=0.001 추천)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()    # 학습 모드 (Dropout 등 활성화)
    total_loss = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 1. 기울기 초기화 (매 배치마다 반드시)
        optimizer.zero_grad()

        # 2. 순전파
        outputs = model(images)

        # 3. 손실 계산
        loss = criterion(outputs, labels)

        # 4. 역전파 (기울기 계산)
        loss.backward()

        # 5. 가중치 업데이트
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset) * 100
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()     # 평가 모드 (Dropout 비활성화)
    total_loss = 0
    correct = 0

    with torch.no_grad():    # 기울기 계산 비활성화 (메모리, 속도 절약)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset) * 100
    return avg_loss, accuracy

EPOCHS = 5
best_acc = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')

print(f"\n최고 테스트 정확도: {best_acc:.2f}%")


# ------------------------------------------------------------
# 04-5. 예측 결과 시각화
# ------------------------------------------------------------

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

model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = outputs.max(1)

# 처음 12개 결과 시각화
fig, axes = plt.subplots(3, 4, figsize=(10, 7))
for i, ax in enumerate(axes.flat):
    img = images[i].cpu().squeeze()
    label = labels[i].item()
    pred = predicted[i].item()
    ax.imshow(img, cmap='gray')
    color = 'green' if pred == label else 'red'
    ax.set_title(f"정답: {label} / 예측: {pred}", color=color, fontsize=9)
    ax.axis('off')
plt.suptitle("초록: 정답  빨강: 오답", fontsize=12)
plt.tight_layout()
plt.show()
