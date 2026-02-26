# ============================================================
# 주제 8. Gradio로 AI 앱 만들기
# ============================================================
# 이 파일은 교재의 실습 코드를 추출한 것입니다.
# Colab 환경에서 실행하세요: https://colab.research.google.com


# ------------------------------------------------------------
# 01. Gradio 기초
# ------------------------------------------------------------
!pip install gradio -q
import gradio as gr
print(f"Gradio {gr.__version__}")

import gradio as gr

def greet(name):
    return f"안녕하세요, {name}님! AI 앱에 오신 것을 환영합니다."

demo = gr.Interface(
    fn=greet,                  # 실행할 함수
    inputs=gr.Textbox(label="이름을 입력하세요"),  # 입력 컴포넌트
    outputs=gr.Textbox(label="인사"),              # 출력 컴포넌트
    title="첫 번째 Gradio 앱"
)
demo.launch()

import gradio as gr
from PIL import Image
import random

def fake_classifier(image):
    """가짜 분류 함수 (테스트용)"""
    classes = ['고양이', '강아지', '새', '자동차']
    probs = [random.random() for _ in classes]
    total = sum(probs)
    return {c: p/total for c, p in zip(classes, probs)}

demo = gr.Interface(
    fn=fake_classifier,
    inputs=gr.Image(type="pil", label="이미지 업로드"),
    outputs=gr.Label(num_top_classes=3, label="분류 결과"),
    title="이미지 분류 데모",
    description="이미지를 업로드하면 분류 결과를 보여줍니다."
)
demo.launch(share=True)


# ------------------------------------------------------------
# 02. 이미지 분류 앱 구현
# ------------------------------------------------------------
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# ─── 1. 모델 로드 ───────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 클래스 정의
CLASSES = ['비행기', '자동차', '새', '고양이', '사슴',
           '개', '개구리', '말', '배', '트럭']

# 모델 아키텍처 재정의 (주제 3에서 정의한 CIFAR_CNN 클래스를 그대로 복사/임포트)
# from cifar_model import CIFAR_CNN  # 또는 아래와 같이 직접 정의
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 256)
        self.fc2   = nn.Linear(256, 10)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

model = CIFAR_CNN().to(device)

# 저장된 가중치 로드
model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
model.eval()

# ─── 2. 전처리 파이프라인 ──────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# ─── 3. 추론 함수 ──────────────────────────────────
def classify_image(image):
    """Gradio에서 호출되는 추론 함수"""
    # PIL Image → 텐서
    tensor = preprocess(image).unsqueeze(0).to(device)  # [1, 3, 32, 32]

    # 추론
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]  # 확률로 변환

    # 딕셔너리 형태로 반환 {클래스명: 확률}
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

# ─── 4. Gradio 앱 구성 ─────────────────────────────
with gr.Blocks(title="CIFAR-10 분류기") as demo:
    gr.Markdown("# CIFAR-10 이미지 분류기")
    gr.Markdown("이미지를 업로드하면 10가지 클래스 중 하나로 분류합니다.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="이미지 업로드")
            classify_btn = gr.Button("분류하기", variant="primary")

        with gr.Column():
            label_output = gr.Label(
                num_top_classes=5,
                label="분류 결과 (상위 5개)"
            )

    classify_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=label_output
    )

    # 예시 이미지 (직접 업로드하거나 examples 폴더 준비)
    gr.Examples(
        examples=[['sample_cat.jpg'], ['sample_dog.jpg']],
        inputs=image_input
    )

demo.launch(share=True)

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="이미지"),
    outputs=gr.Label(num_top_classes=5, label="분류 결과"),
    title="CIFAR-10 실시간 분류기",
)
demo.launch(share=True)


# ------------------------------------------------------------
# 03. Colab에서 Gradio 배포
# ------------------------------------------------------------
demo.launch(
    share=True,          # 공개 URL 생성 (72시간 유효)
    debug=True,          # 에러 메시지 표시
    show_error=True      # 오류 시 UI에 표시
)

with gr.Blocks(title="꽃 분류 AI 앱", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌸 꽃 종류 분류 AI
    **사용법**: 꽃 사진을 업로드하면 5종류의 꽃 중 하나로 분류합니다.

    **분류 가능한 꽃**: 장미 🌹 / 해바라기 🌻 / 튤립 🌷 / 민들레 / 데이지
    """)

    with gr.Tab("분류"):
        with gr.Row():
            img_in = gr.Image(type="pil", label="꽃 사진 업로드")
            lbl_out = gr.Label(num_top_classes=5, label="분류 결과")
        btn = gr.Button("분류 시작!", variant="primary")
        btn.click(fn=classify_image, inputs=img_in, outputs=lbl_out)

    with gr.Tab("모델 정보"):
        gr.Markdown("""
        ## 모델 정보
        - **아키텍처**: ResNet18 (전이학습)
        - **학습 데이터**: Flowers Recognition (Kaggle)
        - **학습 에포크**: 20
        - **테스트 정확도**: 93.2%

        ## 팀 정보
        - **팀명**: AI 꽃미남팀
        - **팀원**: 홍길동, 이순신, 강감찬
        """)

demo.launch(share=True)

# 기존 실행 중인 Gradio 앱 종료
import gradio as gr
gr.close_all()

# 또는 특정 demo 종료
demo.close()

# requirements.txt
torch
torchvision
gradio
Pillow
