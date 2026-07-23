# vast 실행 런북 — 군함 탐지 (YOLO11)

> 전제: RTX 3090 인스턴스, tmux 안에서 실행(§5-4 SSH 끊김 대비). 끝나면 인스턴스 Destroy(§5-5).
> 함정 노트는 상위 `스모크테스트_유의사항.md` 참고.

## 0. 인스턴스 준비물 2개
- **AI Hub 원본 zip** (Google Drive): 이미지 `TS_*.zip` 포함.
- **labels_train.tgz**: stem별 라벨 JSON 41,972개 (로컬 `../data/labels_train`을 tar). Drive에 함께 올리거나 scp.

## 1. 셋업
```bash
cd /workspace
git clone https://github.com/josh-min99/YOLO-pipeline.git
cd YOLO-pipeline
pip install ultralytics gdown
```

## 2. 데이터 내려받기 & 풀기
```bash
# (a) 라벨
gdown <labels_train.tgz FILE_ID> -O labels_train.tgz
mkdir -p labels_train && tar -xzf labels_train.tgz -C labels_train --strip-components=1

# (b) 이미지 zip 번들 (AI Hub 원본)
gdown 1gF4Bjgd7MORlyxnXtPl6SC1Lzl-3yNKS -O aihub.zip
mkdir -p aihub && unzip -q aihub.zip -d aihub
#  -> aihub 안에서 TS_*.zip 위치 확인:  find aihub -name 'TS_*.zip'
#     (개방데이터에 Validation 폴더가 함께 있으면 그 TS_도 포함해 추출 가능)
```

## 3. 이미지 추출 (flat stem.jpg) + YOLO 데이터셋 빌드
```bash
python scripts/extract_images.py --zip-dir "$(dirname $(find aihub -name 'TS_*.zip' | head -1))" \
    --out datasets/marine_frames
# ~40k 프레임, 손상 ~1% 스킵 로그 확인

python scripts/json_to_yolo.py \
    --labels-dir labels_train \
    --images-dir datasets/marine_frames \
    --splits-dir splits \
    --out datasets/marine --link
# [train] frames=... boxes=... / [val] frames=... boxes=...  (img_missing=0 이어야 정상)
```
`configs/marine.yaml`의 `path`가 `../datasets/marine`이면 repo 루트 기준 맞음(필요시 절대경로로).

## 4. 학습 (tmux 안에서)
```bash
tmux new -s train
bash scripts/train.sh yolo11s.pt 1280 100 16
# Ctrl+B D 로 detach. 재접속: tmux attach -t train
```

## 5. 평가 / 확인
```bash
# best 가중치로 val 재평가 (군함=class 2 mAP 확인)
yolo detect val model=runs/detect/marine_*/weights/best.pt data=configs/marine.yaml imgsz=1280

# 몇 장 예측 시각화(bbox 눈으로 확인)
yolo detect predict model=runs/detect/marine_*/weights/best.pt \
    source=datasets/marine/images/val save=True conf=0.25
```

## 6. 결과 회수 후 Destroy
- `runs/detect/marine_*/weights/best.pt` 를 로컬/Drive로 백업.
- `results.csv`, PR/컨퓨전 이미지도 회수.
- 인스턴스 **Destroy** (과금 정지).

## 다음(예정)
- frame-level AUC로 재구성 VAD(60%)와 같은 축에서 비교(프레임 점수 = 군함 conf max).
- 실시간 데모용 FPS 측정(yolo11n vs s, imgsz 스윕). W1처럼 batch=1 + sync.
- open-set 헤드(이상 정의 확정 후).
