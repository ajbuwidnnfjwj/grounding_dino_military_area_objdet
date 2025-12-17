import torch
from PIL import Image
from transformers import ViTImageProcessor
from vit.vit_config import ViTEnvClassifier
from vit.vit_build_data import processor # 모델 정의 파일 임포트

# --- 1. 라벨 맵핑 (사람이 알아볼 수 있게 변환) ---
# 학습 때 -1을 해서 0부터 시작하도록 만들었으므로,
# 여기서는 인덱스 0이 원본 데이터의 1(첫 번째 값)에 해당합니다.
LABEL_MAP = {
    "season": {0: "하절기 (Summer)", 1: "동절기 (Winter)"},
    "night":  {0: "주간 (Day)", 1: "야간 (Night)"},
    "weather": {
        0: "맑음 (Clear)", 1: "구름조금", 2: "구름많음", 
        3: "흐림", 4: "비", 5: "눈/비", 6: "눈" 
        # (실제 데이터셋 정의에 맞춰 수정 필요)
    },
    "wave": {
        0: "파고 1단계 (0~0.5m)", 1: "파고 2단계 (0.5~1.0m)", 
        2: "파고 3단계 (1.0~1.5m)", 3: "파고 4단계 (1.5~2.0m)",
        4: "파고 5단계 (2.0~2.5m)", 5: "파고 6단계 (2.5~3.0m)",
        6: "파고 7단계 (3.0m 이상)"
    }
}

def predict_image(image_path, model_path = "vit/vit_env.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Prediction using device: {device}")

    # --- 2. 모델 로드 ---
    # 구조를 먼저 만들고, 저장된 가중치(state_dict)를 덮어씌웁니다.
    model = ViTEnvClassifier()
    
    # map_location은 CPU 환경에서도 로드되도록 안전장치를 둡니다.
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()  # 평가 모드로 전환 (Dropout, BatchNorm 등을 고정)

    # --- 3. 이미지 전처리 ---    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"이미지를 열 수 없습니다: {e}")
        return

    # 픽셀 값을 텐서로 변환
    inputs = processor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # --- 4. 추론 (Inference) ---
    with torch.no_grad():  # 기울기 계산 비활성화 (메모리 절약)
        outputs = model(pixel_values)

    # --- 5. 결과 해석 ---
    print(f"\n[{image_path}] 분석 결과:")
    print("-" * 30)
    
    results = {}
    for task_name, logits in outputs.items():
        # Softmax를 거쳐 확률이 가장 높은 인덱스를 찾음
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100  # 확신도(%)
        
        # 라벨 맵핑을 통해 텍스트로 변환
        label_text = LABEL_MAP[task_name].get(pred_idx, f"Unknown({pred_idx})")
        
        print(f"{task_name.upper():<8} : {label_text} ({confidence:.1f}%)")
        results[task_name] = label_text

    return results

# --- 실행부 ---
if __name__ == "__main__":
    # 테스트할 이미지 경로와 학습된 모델 경로를 지정하세요.
    TEST_IMAGE = "data/test/image/VS_EO_SU_DT/EO_SU_DT_W1_H1_E1B1C5_0001.jpg"  # 실제 경로로 수정
    MODEL_PATH = "vit_env.pth"            # 저장된 모델 파일명
    
    print(predict_image(TEST_IMAGE, MODEL_PATH))