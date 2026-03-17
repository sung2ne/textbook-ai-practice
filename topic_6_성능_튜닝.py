# ============================================================
# 주제 6. 성능 튜닝
# ============================================================
# 이 파일은 교재의 실습 코드를 추출한 것입니다.
# Colab 환경에서 실행하세요: https://colab.research.google.com


# ------------------------------------------------------------
# 01. 학습 곡선 분석
# ------------------------------------------------------------
import torch

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

history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [],  'val_acc': []
}

EPOCHS = 30
for epoch in range(1, EPOCHS + 1):
    # 학습
    model.train()
    tr_loss, tr_correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        tr_correct += outputs.max(1)[1].eq(labels).sum().item()

    # 검증
    model.eval()
    vl_loss, vl_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            vl_loss += criterion(outputs, labels).item()
            vl_correct += outputs.max(1)[1].eq(labels).sum().item()

    history['train_loss'].append(tr_loss / len(train_loader))
    history['val_loss'].append(vl_loss / len(val_loader))
    history['train_acc'].append(tr_correct / len(train_loader.dataset) * 100)
    history['val_acc'].append(vl_correct / len(val_loader.dataset) * 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 손실 곡선
axes[0].plot(history['train_loss'], label='학습 손실', color='blue')
axes[0].plot(history['val_loss'],   label='검증 손실', color='orange')
axes[0].set_title('손실 곡선')
axes[0].set_xlabel('에포크')
axes[0].set_ylabel('Loss')
axes[0].legend()

# 정확도 곡선
axes[1].plot(history['train_acc'], label='학습 정확도', color='blue')
axes[1].plot(history['val_acc'],   label='검증 정확도', color='orange')
axes[1].set_title('정확도 곡선')
axes[1].set_xlabel('에포크')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()

plt.tight_layout()
plt.show()

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

optimizer = optim.Adam(model.parameters(), lr=0.001)

# CosineAnnealingLR: 코사인 곡선으로 학습률을 감소
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

for epoch in range(EPOCHS):
    # 학습 ...
    scheduler.step()  # 에포크 후 학습률 업데이트
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}: lr = {current_lr:.6f}")


# ------------------------------------------------------------
# 02. 하이퍼파라미터 조정
# ------------------------------------------------------------
import itertools

# 탐색할 하이퍼파라미터 조합
param_grid = {
    'lr':         [0.01, 0.001, 0.0001],
    'batch_size': [64, 128],
    'dropout':    [0.3, 0.5]
}

results = []

for lr, bs, dp in itertools.product(
    param_grid['lr'],
    param_grid['batch_size'],
    param_grid['dropout']
):
    print(f"\n실험: lr={lr}, batch_size={bs}, dropout={dp}")

    # 데이터 로더 재생성
    loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    # 모델 재생성
    class QuickCNN(nn.Module):
        def __init__(self, dropout):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64*8*8, 256), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 10)
            )
        def forward(self, x): return self.classifier(self.features(x))

    model_exp = QuickCNN(dp).to(device)
    opt = optim.Adam(model_exp.parameters(), lr=lr)

    # 빠른 검증 (5 에포크만)
    for epoch in range(5):
        model_exp.train()
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            loss = criterion(model_exp(imgs), lbls)
            loss.backward(); opt.step()

    model_exp.eval()
    correct = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            correct += model_exp(imgs).max(1)[1].eq(lbls).sum().item()
    acc = correct / len(val_dataset) * 100

    results.append({'lr': lr, 'bs': bs, 'dp': dp, 'acc': acc})
    print(f"  검증 정확도: {acc:.2f}%")

# 최적 조합 출력
best = max(results, key=lambda x: x['acc'])
print(f"\n최적 조합: {best}")

import pandas as pd

df = pd.DataFrame(results)
df = df.sort_values('acc', ascending=False)
print(df.to_string(index=False))


# ------------------------------------------------------------
# 03. 모델 비교와 선택
# ------------------------------------------------------------
import torch

# 최고 성능 모델 저장
best_acc = 0
for epoch in range(EPOCHS):
    # ... 학습 및 평가 ...
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_model.pth')
        print(f"  ✓ 모델 저장 (val_acc: {val_acc:.2f}%)")

# 모델 로드
checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)  # 딕셔너리 포함
model.load_state_dict(checkpoint['model_state_dict'])
print(f"로드된 모델 - epoch: {checkpoint['epoch']}, acc: {checkpoint['val_acc']:.2f}%")

import time

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, device, input_size=(1, 3, 32, 32), repeats=100):
    model.eval()
    dummy = torch.rand(*input_size).to(device)
    # 워밍업
    for _ in range(10):
        _ = model(dummy)
    # 측정
    start = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy)
    elapsed = (time.time() - start) / repeats * 1000
    return elapsed  # ms

# 여러 모델 비교
models_to_compare = {
    'SimpleNet (MLP)': SimpleNet(),
    'CIFAR_CNN': CIFAR_CNN(),
}

print(f"{'모델':<20} {'파라미터':>12} {'추론시간(ms)':>14} {'정확도':>8}")
print("-" * 60)
for name, m in models_to_compare.items():
    m = m.to(device)
    params = count_params(m)
    inf_time = measure_inference_time(m, device)
    # 실제 정확도는 학습 후 측정
    print(f"{name:<20} {params:>12,} {inf_time:>12.2f}ms  {'측정 필요':>8}")
