# ============================================================
# 주제 1. 개발 환경 설정
# ============================================================
# 이 파일은 교재의 실습 코드를 추출한 것입니다.
# Colab 환경에서 실행하세요: https://colab.research.google.com


# ------------------------------------------------------------
# 01. Google Colab 시작하기
# ------------------------------------------------------------
import torch

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# GPU 정보 출력
if torch.cuda.is_available():
    print(f"GPU 모델: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Gradio 설치 (AI 앱 UI용) — 주제 8에서 사용
!pip install gradio -q

import gradio as gr
print(f"Gradio 버전: {gr.__version__}")

from google.colab import drive

# Google Drive 마운트
drive.mount('/content/drive')

# Drive 내 폴더 확인
import os
print(os.listdir('/content/drive/MyDrive'))

# 해결 1: 배치 크기 줄이기
batch_size = 64  # 128 → 64 → 32 순서로 줄여보기

# 해결 2: 캐시 비우기
import torch
torch.cuda.empty_cache()

# 해결 3: 사용 후 변수 삭제
del large_tensor
torch.cuda.empty_cache()

# 해결: 노트북 맨 위에 항상 실행할 셀 모아두기
!pip install gradio -q
from google.colab import drive
drive.mount('/content/drive')

import os

# 현재 위치 확인
print(os.getcwd())           # /content
print(os.listdir('/content')) # 파일 목록 확인

# Drive에 저장된 경우
model_path = '/content/drive/MyDrive/AI_Project/best_model.pth'
print(os.path.exists(model_path))  # True이면 파일 존재


# ------------------------------------------------------------
# 02. Kaggle Notebooks 시작하기
# ------------------------------------------------------------
import os

# 추가된 데이터셋 목록 확인
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5]:  # 처음 5개만 출력
        print(os.path.join(dirname, filename))

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

if torch.cuda.is_available():
    print(f"GPU 수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


# ------------------------------------------------------------
# 03. PyTorch 첫 만남
# ------------------------------------------------------------
import torch
import numpy as np

# 1차원 텐서 (벡터)
a = torch.tensor([1.0, 2.0, 3.0])
print(f"1D 텐서: {a}")
print(f"shape: {a.shape}, dtype: {a.dtype}")

# 2차원 텐서 (행렬)
b = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float32)
print(f"\n2D 텐서:\n{b}")
print(f"shape: {b.shape}")

# 0으로 채운 텐서
zeros = torch.zeros(3, 4)
print(f"\n3×4 영행렬:\n{zeros}")

# 랜덤 텐서 (0~1 균등분포)
rand = torch.rand(2, 3)
print(f"\n2×3 랜덤 텐서:\n{rand}")

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 기본 연산
print(f"덧셈: {x + y}")
print(f"곱셈: {x * y}")
print(f"내적: {torch.dot(x, y)}")

# 행렬 곱
A = torch.rand(3, 4)
B = torch.rand(4, 2)
C = torch.matmul(A, B)  # 또는 A @ B
print(f"\n행렬 곱 결과 shape: {C.shape}")  # [3, 2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CPU 텐서 생성
x_cpu = torch.rand(1000, 1000)

# GPU로 이동
x_gpu = x_cpu.to(device)
print(f"텐서 위치: {x_gpu.device}")

# GPU에서 직접 생성
y_gpu = torch.rand(1000, 1000, device=device)

# GPU 연산 (매우 빠름)
result = torch.matmul(x_gpu, y_gpu)
print(f"결과 shape: {result.shape}")

# requires_grad=True: 이 텐서의 연산 기록을 추적
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 순전파
x = torch.tensor(3.0)
y = w * x + b   # y = 2*3 + 1 = 7

# 역전파 (기울기 계산)
y.backward()

# 기울기 확인
print(f"∂y/∂w = {w.grad}")  # x = 3.0
print(f"∂y/∂b = {b.grad}")  # 1.0

import numpy as np

# NumPy → PyTorch
np_array = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(np_array)
print(f"NumPy → Tensor: {tensor}")

# PyTorch → NumPy
tensor2 = torch.tensor([4.0, 5.0, 6.0])
np_array2 = tensor2.numpy()
print(f"Tensor → NumPy: {np_array2}")

# GPU 텐서를 NumPy로 변환할 때는 CPU로 먼저 이동
gpu_tensor = torch.rand(3, device=device)
np_from_gpu = gpu_tensor.cpu().numpy()
print(f"GPU Tensor → NumPy: {np_from_gpu}")
