# ============================================================
# 주제 11. 프로젝트 완성 가이드
# ============================================================
# 이 파일은 교재의 실습 코드를 추출한 것입니다.
# Colab 환경에서 실행하세요: https://colab.research.google.com


# ------------------------------------------------------------
# 01. Gradio 앱 마무리
# ------------------------------------------------------------
import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ─── 모델 초기화 ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ['데이지', '민들레', '장미', '해바라기', '튤립']
EMOJI   = ['🌼', '🌿', '🌹', '🌻', '🌷']

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, len(CLASSES)))
model.load_state_dict(torch.load('efficientnet_flowers.pth', map_location=device, weights_only=True))
model.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─── 추론 함수 ──────────────────────────────────────
def classify(image):
    if image is None:
        return None
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return {f"{EMOJI[i]} {CLASSES[i]}": float(probs[i]) for i in range(len(CLASSES))}

# ─── Gradio UI ──────────────────────────────────────
css = ".gradio-container { font-family: 'Noto Sans KR', sans-serif; }"

with gr.Blocks(
    title="꽃 분류 AI",
    theme=gr.themes.Soft(primary_hue="pink"),
    css=css
) as demo:
    gr.Markdown("""
    # 🌸 꽃 종류 분류 AI
    **팀명**: AI 꽃미남팀 | **모델**: EfficientNet-B0 (전이학습) | **정확도**: 93.2%

    꽃 사진을 업로드하거나 예시 이미지를 클릭하세요.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="꽃 사진 업로드",
                                  height=300)
            classify_btn = gr.Button("분류하기 🔍", variant="primary", size="lg")

        with gr.Column(scale=1):
            label_output = gr.Label(
                num_top_classes=5,
                label="분류 결과"
            )
            gr.Markdown("*확률이 높을수록 해당 꽃일 가능성이 높습니다.*")

    # 예시 이미지
    gr.Examples(
        examples=['examples/rose.jpg', 'examples/sunflower.jpg', 'examples/daisy.jpg'],
        inputs=img_input,
        label="예시 이미지"
    )

    classify_btn.click(fn=classify, inputs=img_input, outputs=label_output)

demo.launch(share=True, debug=False)

def classify_safe(image):
    """에러 처리가 포함된 안전한 추론 함수"""
    if image is None:
        return {"오류": 1.0}  # 빈 입력 처리
    try:
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    except Exception as e:
        print(f"추론 오류: {e}")
        return {"처리 중 오류 발생": 1.0}


# ------------------------------------------------------------
# 02. 발표 준비 가이드
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

# 발표용 결과 정리 코드
import matplotlib.pyplot as plt
import numpy as np

# 1차 vs 2차 프로젝트 성능 비교
models = ['직접 만든 CNN\n(1차)', 'ResNet18\n전이학습 (2차)', 'EfficientNet-B0\n전이학습 (최종)']
accuracies = [75.3, 91.8, 93.2]
colors = ['#FF6B6B', '#FFA07A', '#4CAF50']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, accuracies, color=colors, width=0.4)
ax.set_ylim(60, 100)
ax.set_ylabel('테스트 정확도 (%)', fontsize=12)
ax.set_title('프로젝트 성능 향상 과정', fontsize=14)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150)
plt.show()


# ------------------------------------------------------------
# 03. 포트폴리오 정리
# ------------------------------------------------------------