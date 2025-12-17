# vit, grounding dino, vision llm full path

grounding dino의 호환성이 박살난 관계로 환경 설정이 조금 복잡함.
python 3.9(맥북에 기본 탑재), venv 환경에서 구축

mmdetection 깃 저장소 클론 <br>
```git clone https://github.com/open-mmlab/mmdetection.git```

mmdetection/requirements/multimodal.txt의 의존성 설치 <br>
```pip install -r mmdetection/requirements/multimodal.txt```

torch버전을 2.5로 내려야함. 보안정책이 2.6부터 강화되어서 복잡해짐 <br>
```pip uninstll torch, torchvision``` <br>
```pip install torch==2.5.0 torchvision==0.20.1```

mmdetection 구동에 필요한 라이브러리들 설치 <br>
```pip install -U openmim``` <br>
```mim install mmengine "mmcv>=2.0.0" mmdet```

여기까지 하면 의존성 설치 완료, 경로 잘 설정해서 vllm 파일 실행하면 댐
