import torch
import os

# -------------------------------------------------------------------------
# [2] 설정 변수 (이 부분만 수정해서 쓰세요)
# -------------------------------------------------------------------------
# Config와 Checkpoint 경로 (사용자 경로에 맞춤)
config_file = 'grounding_dino_config.py'
checkpoint_file = 'grounding_dino.pth'

# 테스트할 이미지 경로
image_path = 'data/test/image/VS_EO_SU_DT/EO_SU_DT_W1_H1_E1B1C5_0001.jpg'

# 찾고 싶은 텍스트 프롬프트 (점(.)으로 구분)
text_prompt = "어선 . 군함 . 상선 . 고정익 유인기 . 회전익 유인기 . 무인항공기 . 새 . 삐라 . 오물폭탄 ."

# 결과 이미지를 저장할 경로
output_file = 'result_prediction.jpg'

# 신뢰도 임계값 (0.3 이하는 버림)
score_thr = 0.3

# -------------------------------------------------------------------------
# [3] 디바이스 설정 (MacBook 최적화)
# -------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

command = "python mmdetection/demo/image_demo.py " + image_path + " " + config_file + " " \
    + "--weights " + checkpoint_file + " " \
    + "--texts \"어선 . 군함 . 상선 . 고정익 유인기 . 회전익 유인기 . 무인항공기 . 새 . 삐라 . 오물폭탄 .\" " \
    + "--device cpu " \
    + "--pred-score-thr " + str(score_thr) + " " \
    + "--palette random "
    

print(command)
os.system(command)