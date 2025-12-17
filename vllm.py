import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

import json
import subprocess

from vit import predict_image

image_path = "data/test/image/VS_EO_SU_DT/EO_SU_DT_W1_H1_E1B1C5_0001.jpg"
image_name = "EO_SU_DT_W1_H1_E1B1C5_0001"
image = Image.open(image_path).convert("RGB")

# vit 모델 예측
env_predicted = predict_image(image_path)

# grounding dino 모델 예측
config_file = 'grounding_dino_config.py'
checkpoint_file = 'grounding_dino.pth'
score_thr = 0.3

cmd = [
    "python", "mmdetection/demo/image_demo.py",
    image_path,
    config_file,
    "--weights", checkpoint_file,
    "--texts", "어선 . 군함 . 상선 . 고정익 유인기 . 회전익 유인기 . 무인항공기 . 새 . 삐라 . 오물폭탄 .",
    "--device", "cpu",
    "--pred-score-thr", str(score_thr),
    "--palette", "random"
]

result = subprocess.run(
    cmd,
    check=True   # 실패 시 예외 발생
)

print("grounding dino complete")

with open("./outputs/preds/"+image_name+".json", "r", encoding="utf8") as f:
    objects = json.load(f)
    bboxes = objects['bboxes']
    scores = objects['scores']
    labels = objects['labels']

    scores = [s for s in scores if s >= 0.3]
    bboxes = bboxes[0:len(scores)]
    labels = labels[0:len(scores)]

objects_detected = []
for bbox, label in zip(bboxes, labels):
    objects_detected.append(
        {
            "class" : label,
            "bounding_box": list(map(int, bbox))
        }
    )
          
env_example = {
    'season': '하절기 (Summer)', 
    'night': '주간 (Day)', 
    'weather': '맑음 (Clear)', 
    'wave': '파고 1단계 (0~0.5m)'
}
obj_example = [
    {
      "class": 4,
      "bounding_box": [
        15,
        165,
        1189,
        159
      ]
    },
    {
      "class": 2,
      "bounding_box": [
        51,
        26,
        1100,
        222
      ]
    }
  ]


text_prompt = f"""사진과 주어진 정보들을 보고 한줄 캡션을 생성해줘. 주어지는 정보는 환경정보와 객체정보가 있어
    캡션을 생성하는데 주어지는 환경정보 예시는 다음과 같아.
    {env_example}
    캡션을 생성하는데 주어지는 객체정보 예시는 다음과 같아. 객체가 많은 경우가 있을 수 있어서 리스트의 형식으로 주어질거야.
    {obj_example}

    각 객체에 대해서 class는 객체가 어떤 물체인지 나타내는 정보야. 다음과 같이 맵핑하면 돼
    class:
    선박 = 1, 항공기 = 2, 동물 = 3, 기타 = 4

    bbox는 물체의 bounding box야. [w, h, x, y] 형식으로 나타냈으니 헷갈리지 않도록 주의해

    주어진 이미지에 대한 환경정보는 {env_predicted} 이고, 객체정보는 {objects_detected}야

    출력은 단 한개의 완성된 문장으로 답변해야해"""

model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("mps")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text_prompt},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16).to("mps")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )

print(processor.decode(output[0][2:], skip_special_tokens=True))
