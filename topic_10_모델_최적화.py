# ============================================================
# 주제 10. 모델 최적화
# ============================================================
# 이 파일은 교재의 실습 코드를 추출한 것입니다.
# Colab 환경에서 실행하세요: https://colab.research.google.com


# ------------------------------------------------------------
# 01. 학습률 스케줄링
# ------------------------------------------------------------
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 1. StepLR: step_size 에포크마다 gamma 배로 감소
scheduler_step = StepLR(optimizer, step_size=7, gamma=0.1)
# 결과: 0.001 → 0.0001 (7에포크마다)

# 2. CosineAnnealingLR: 코사인 곡선으로 부드럽게 감소
scheduler_cos = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
# 결과: 0.001 → 1e-6 (30에포크에 걸쳐 코사인 감소)

# 3. ReduceLROnPlateau: 검증 손실이 개선되지 않으면 감소
scheduler_plateau = ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)
# 결과: 검증 손실이 3에포크 동안 개선 없으면 0.5배로 감소
# ※ verbose=True는 PyTorch 2.2+에서 deprecated. 학습률 변화는 optimizer.param_groups[0]['lr']로 직접 출력

# CosineAnnealingLR 사용 예시
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

lr_history = []
for epoch in range(EPOCHS):
    # 학습 ...
    scheduler.step()  # 에포크마다 호출
    lr_history.append(optimizer.param_groups[0]['lr'])

# Colab 한글 폰트 설정
import subprocess
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

subprocess.run(['apt-get', '-qq', 'install', '-y', 'fonts-nanum'],
               capture_output=True, check=True)

fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 학습률 변화 시각화
import matplotlib.pyplot as plt
plt.plot(lr_history)
plt.title('CosineAnnealingLR 학습률 변화')
plt.xlabel('에포크')
plt.ylabel('학습률')
plt.yscale('log')
plt.show()

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

for epoch in range(EPOCHS):
    # 학습 ...
    val_loss = evaluate(...)  # 검증 손실 계산

    # val_loss를 전달 (에포크 후 호출)
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}: lr = {current_lr:.6f}")


# ------------------------------------------------------------
# 02. 모델 저장과 로드
# ------------------------------------------------------------
import torch

# 방법 1: state_dict만 저장 (권장)
# 모델 아키텍처는 코드에 있고, 가중치만 저장
torch.save(model.state_dict(), 'model_weights.pth')

# 로드: 아키텍처 먼저 생성 후 가중치 로드
model = MyModel()
model.load_state_dict(torch.load('model_weights.pth', map_location='cpu', weights_only=True))
model.eval()

# 방법 2: 전체 모델 저장 (편리하지만 이식성 낮음)
torch.save(model, 'full_model.pth')
model = torch.load('full_model.pth', map_location='cpu', weights_only=False)

import os

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
    }, path)

def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)  # optimizer/scheduler 딕셔너리 포함
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt['val_acc']


# 체크포인트에서 재개
START_EPOCH = 0
if os.path.exists('checkpoint.pth'):
    START_EPOCH, prev_acc = load_checkpoint(model, optimizer, scheduler,
                                            'checkpoint.pth', device)
    print(f"체크포인트 로드: epoch {START_EPOCH}, val_acc {prev_acc:.2f}%")

for epoch in range(START_EPOCH + 1, EPOCHS + 1):
    # 학습 ...
    save_checkpoint(model, optimizer, scheduler, epoch, val_acc, 'checkpoint.pth')

import os
from google.colab import drive
drive.mount('/content/drive')

save_path = '/content/drive/MyDrive/AI_Project/models/best_model.pth'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 최고 성능 모델 Drive에 저장
best_acc = 0
for epoch in range(EPOCHS):
    # 학습 + 평가 ...
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"  Drive에 저장: {val_acc:.2f}%")
