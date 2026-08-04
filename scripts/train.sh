#!/usr/bin/env bash
# vast에서 YOLO11 군함 탐지 학습. tmux 안에서 실행할 것(§5-4).
# 사용: bash scripts/train.sh [model] [imgsz] [epochs] [batch]
set -e
MODEL=${1:-yolo11s.pt}   # 실시간 데모용 경량. 더 가볍게: yolo11n.pt
# 모델 크기 비교 run(W9): yolo11m(20.1M/68.3GFLOPs) vs 기본 yolo11s(9.4M/21.4GFLOPs).
#   bash scripts/train.sh yolo11m.pt 1280 100 48
# 상선이 새 지점에서 0.913으로 약한 것이 용량 문제인지 데이터 문제인지 가르는 실험.
# batch는 GPU 4장 합. 11m은 s보다 무거우므로 48(장당 12) 권장.
IMGSZ=${2:-1280}         # 군함이 작음(native 1920서 ~50-100px) → 고해상 유리. 낮은 recall이면 1536.
EPOCHS=${3:-100}
BATCH=${4:-16}           # 3090 24GB. OOM이면 낮추기(또는 -1로 자동).

yolo detect train \
  model="$MODEL" \
  data=configs/marine.yaml \
  imgsz="$IMGSZ" epochs="$EPOCHS" batch="$BATCH" \
  device=0 workers=8 \
  project=runs name="marine_${MODEL%.pt}_${IMGSZ}" \
  patience=30 \
  cache=False
# 결과: runs/detect/marine_*/  (weights/best.pt, results.csv, PR curve 등)
# 군함 성능은 val 로그의 class 2(warship) mAP50 / mAP50-95 를 볼 것.
