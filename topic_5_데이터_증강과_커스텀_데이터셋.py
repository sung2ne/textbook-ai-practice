# ============================================================
# 주제 5. 데이터 증강과 커스텀 데이터셋
# ============================================================
# 이 파일은 교재의 실습 코드를 추출한 것입니다.
# Colab 환경에서 실행하세요: https://colab.research.google.com


# ------------------------------------------------------------
# 01. Transforms와 Augmentation
# ------------------------------------------------------------
from torchvision import transforms

# 학습용 증강 파이프라인 (다양하게 설정)
train_transform = transforms.Compose([
    # 기하학적 변환
    transforms.RandomHorizontalFlip(p=0.5),    # 50% 확률로 좌우 반전
    transforms.RandomVerticalFlip(p=0.1),      # 10% 확률로 상하 반전
    transforms.RandomRotation(degrees=15),      # ±15도 랜덤 회전
    transforms.RandomCrop(32, padding=4),       # 패딩 후 랜덤 크롭
    transforms.RandomResizedCrop(224),          # 랜덤 크기+위치 크롭

    # 색상 변환
    transforms.ColorJitter(
        brightness=0.2,   # 밝기 ±20%
        contrast=0.2,     # 대비 ±20%
        saturation=0.2,   # 채도 ±20%
        hue=0.1           # 색조 ±10%
    ),
    transforms.RandomGrayscale(p=0.1),         # 10% 확률로 흑백 변환

    # 필수 변환
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet 정규화값
])

# 테스트용 (증강 없이 크기 조정과 정규화만)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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
from PIL import Image
import torch

# 단일 이미지에 증강 반복 적용
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor()
])

# 테스트 이미지 로드 (CIFAR-10 첫 번째 이미지)
from torchvision.datasets import CIFAR10
base_dataset = CIFAR10('./data', train=True, download=True)
orig_img, label = base_dataset[0]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    aug_img = aug_transform(orig_img)
    ax.imshow(aug_img.permute(1, 2, 0).numpy().clip(0, 1))
    ax.set_title(f"증강 {i+1}", fontsize=9)
    ax.axis('off')
plt.suptitle(f"원본 클래스: {CIFAR10.classes[label]} — 같은 이미지 10가지 증강", fontsize=11)
plt.tight_layout()
plt.show()

import numpy as np

def mixup_data(x, y, alpha=0.2):
    """두 배치 이미지를 알파 비율로 섞는 Mixup"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 학습 루프에서 사용
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    images_mix, y_a, y_b, lam = mixup_data(images, labels)
    outputs = model(images_mix)
    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ------------------------------------------------------------
# 02. 나만의 Dataset 클래스
# ------------------------------------------------------------

# 2.1 데이터셋 다운로드
import urllib.request
import zipfile
import os
import random
import shutil
from PIL import Image, UnidentifiedImageError

url = ("https://download.microsoft.com/download/3/E/1/"
       "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")
urllib.request.urlretrieve(url, "download.zip")
print("다운로드 완료")

with zipfile.ZipFile("download.zip", "r") as zip_ref:
    zip_ref.extractall(".")
print("압축 해제 완료")

removed = 0
for class_dir in ["Cat", "Dog"]:
    folder = os.path.join("PetImages", class_dir)
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        try:
            img = Image.open(fpath)
            img.verify()
        except Exception:
            os.remove(fpath)
            removed += 1
print(f"손상된 이미지 {removed}개 제거")

random.seed(42)
for class_dir in ["Cat", "Dog"]:
    files = [f for f in os.listdir(os.path.join("PetImages", class_dir))
             if f.lower().endswith(("jpg", "jpeg", "png"))]
    random.shuffle(files)
    split = int(len(files) * 0.8)
    for dest, file_list in [("train", files[:split]), ("test", files[split:])]:
        dest_dir = os.path.join(dest, class_dir)
        os.makedirs(dest_dir, exist_ok=True)
        for f in file_list:
            shutil.copyfile(
                os.path.join("PetImages", class_dir, f),
                os.path.join(dest_dir, f)
            )

print(f"학습: {len(os.listdir('train/Cat'))+len(os.listdir('train/Dog'))}장")
print(f"테스트: {len(os.listdir('test/Cat'))+len(os.listdir('test/Dog'))}장")

# 2.2 ImageFolder로 데이터 로드
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ImageFolder('train', transform=train_transform)
test_dataset = ImageFolder('test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"클래스: {train_dataset.classes}")
print(f"클래스 인덱스: {train_dataset.class_to_idx}")
print(f"학습: {len(train_dataset)}장, 테스트: {len(test_dataset)}장")

# 2.3 커스텀 Dataset 클래스
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

# 2.5 데이터셋 통계 계산
def calculate_stats(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total = 0
    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)
        mean += images.mean(2).sum(0)
        std  += images.std(2).sum(0)
        total += batch_size
    mean /= total
    std  /= total
    return mean.tolist(), std.tolist()

raw_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
raw_dataset = ImageFolder('train', transform=raw_transform)
mean, std = calculate_stats(raw_dataset)
print(f"평균: {mean}")
print(f"표준편차: {std}")

# 2.6 모델 학습
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += outputs.max(1)[1].eq(labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset) * 100


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += outputs.max(1)[1].eq(labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset) * 100


model = CatDogCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        torch.save(model.state_dict(), 'catdog_best.pth')

print(f"\n최고 테스트 정확도: {best_acc:.2f}%")

# 2.7 예측 테스트
# Colab 한글 폰트 설정
import subprocess
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

subprocess.run(['apt-get', '-qq', 'install', '-y', 'fonts-nanum'],
               capture_output=True, check=True)

fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

model.load_state_dict(torch.load('catdog_best.pth', weights_only=True))
model.eval()

images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = outputs.max(1)

classes = ['Cat', 'Dog']
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = images[i].cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img * std + mean).permute(1, 2, 0).numpy().clip(0, 1)
    ax.imshow(img)
    color = 'green' if predicted[i] == labels[i] else 'red'
    ax.set_title(f"정답: {classes[labels[i]]} / 예측: {classes[predicted[i]]}",
                 color=color, fontsize=9)
    ax.axis('off')
plt.suptitle("초록: 정답  빨강: 오답", fontsize=12)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 03. Kaggle 데이터셋 활용하기
# ------------------------------------------------------------
# 1. kaggle.json 업로드 (Kaggle 계정 → Settings → Create API Token)
from google.colab import files
files.upload()  # kaggle.json 파일 업로드

# 2. API 키 설정
import os
os.makedirs('/root/.kaggle', exist_ok=True)
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

# 3. kaggle 패키지 설치
!pip install kaggle -q

# 데이터셋 검색
!kaggle datasets list --search "dogs-vs-cats"

# 데이터셋 다운로드
!kaggle datasets download -d salader/dogs-vs-cats -p ./data/dogs-cats --unzip

# 다운로드된 구조 확인
import os
for root, dirs, files in os.walk('./data/dogs-cats'):
    level = root.replace('./data/dogs-cats', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 2:
        for file in files[:3]:
            print(f"{indent}  {file}")

import os

# Kaggle Notebooks에서 데이터셋 경로 확인
data_dir = '/kaggle/input/dogs-vs-cats'
print(os.listdir(data_dir))

# ImageFolder로 바로 로드
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageFolder(root=f'{data_dir}/train', transform=transform)
print(f"클래스: {dataset.classes}")
print(f"이미지 수: {len(dataset)}")

# 1. Kaggle API 설정 (3.1절 참조)
import os
os.makedirs('/root/.kaggle', exist_ok=True)

# kaggle.json 이미 업로드된 상태에서:
!mv kaggle.json /root/.kaggle/ && chmod 600 /root/.kaggle/kaggle.json
!pip install kaggle -q

# 2. Flowers Recognition 다운로드
!kaggle datasets download -d alxmamaev/flowers-recognition -p /content/data --unzip

# 3. 폴더 구조 확인
for cls in sorted(os.listdir('/content/data/flowers')):
    n = len(os.listdir(f'/content/data/flowers/{cls}'))
    print(f"  {cls}: {n}장")

from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageFolder(root='/content/data/flowers', transform=transform)
print(f"클래스: {dataset.classes}")   # ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
print(f"총 이미지: {len(dataset)}장") # 4317장
