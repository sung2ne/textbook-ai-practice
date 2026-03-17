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
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageFolder(root='data/train', transform=transform)
print(f"클래스: {dataset.classes}")       # ['강아지', '고양이']
print(f"클래스 인덱스: {dataset.class_to_idx}")  # {'강아지': 0, '고양이': 1}
print(f"전체 이미지: {len(dataset)}장")

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class CustomImageDataset(Dataset):
    """
    CSV 파일과 이미지 폴더를 읽는 커스텀 Dataset

    CSV 형식:
    filename,label
    img001.jpg,0
    img002.jpg,1
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """데이터셋의 전체 크기를 반환"""
        return len(self.df)

    def __getitem__(self, idx):
        """idx번째 데이터를 반환"""
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label


# 사용 예시
dataset = CustomImageDataset(
    csv_file='data/labels.csv',
    img_dir='data/images/',
    transform=transform
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

from torch.utils.data import DataLoader
import torch

def calculate_stats(dataset):
    """데이터셋의 채널별 평균과 표준편차를 계산"""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total = 0

    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)  # [B, C, H*W]
        mean += images.mean(2).sum(0)
        std  += images.std(2).sum(0)
        total += batch_size

    mean /= total
    std  /= total
    return mean.tolist(), std.tolist()

# 사용 예시 (transform에 정규화 없이)
raw_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
raw_dataset = CustomImageDataset('data/labels.csv', 'data/images/', raw_transform)
mean, std = calculate_stats(raw_dataset)
print(f"평균: {mean}")
print(f"표준편차: {std}")


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
