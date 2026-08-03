#!/bin/bash
# W8 — 합성 야간/IR 사전학습 -> 실데이터 파인튜닝 (2단계).
#
# 왜 2단계인가:
#   71858 실데이터 train ~28,800프레임 vs 71856 합성(야간EO+IR) 57,534프레임.
#   합성이 2배라 한 번에 섞으면 학습 분포가 합성 쪽으로 끌린다. zero-shot 측정에서
#   두 도메인은 군함 mAP50 0.979 대 0.654만큼 떨어져 있다(W7). 사전학습->파인튜닝으로
#   야간 특징만 가져오고 최종 가중치는 실 도메인에 정렬시킨다.
#
# 라벨 공간(옵션 B): 71856도 선박 3클래스만 라벨하고 항공기·새·삐라는 무라벨로 둔다.
#   71858도 사람(3)·조류(4)를 이미 그렇게 버리고 있어(json_to_yolo.py) 규약이 일치하고,
#   군함 박스를 100% 보존한다. 무라벨 항공기는 hard negative로 유익하다.
#
# 🔴 판정은 2단계 결과를 W5 실데이터 홀드아웃으로 재는 것. 71856 val 점수가 오르는 것은
#    같은 렌더러 안의 성적이라 성과가 아니다(W7 §5, 유의사항 §16-8).
#
# 사용:
#   bash scripts/train_w8.sh prep     # zip -> YOLO 변환 (합성)
#   bash scripts/train_w8.sh stage1   # 합성 사전학습
#   bash scripts/train_w8.sh stage2   # 실데이터 파인튜닝
#   bash scripts/train_w8.sh eval     # W5 홀드아웃 + 71856 조건별
set -e

STEP=${1:?prep|stage1|stage2|eval}
SYNTH_ZIPS=${SYNTH_ZIPS:-/workspace/synth_train}      # TS_*.zip / TL_*.zip
SYNTH_DS=${SYNTH_DS:-datasets/synth_train}            # 변환 결과
REAL_DATA=${REAL_DATA:-configs/marine.yaml}           # 71858 (지점 홀드아웃 분할)
IMGSZ=${IMGSZ:-1280}
DEVICE=${DEVICE:-0,1,2,3}
# 🔴 ultralytics에서 batch는 4장 전체 합. 단일 GPU 16 기준이면 4장은 64.
BATCH=${BATCH:-64}
WORKERS=${WORKERS:-16}

case "$STEP" in

prep)
  # 합성 전체 이미지 + 선박만 라벨. --ships-only 를 주지 않는 것이 옵션 B.
  # 선박 0장 이미지(항공기/새만 있는 것)는 빈 .txt = 배경 negative로 들어간다.
  python scripts/synth71856_to_yolo.py \
      --vs-dir "$SYNTH_ZIPS" --vl-dir "$SYNTH_ZIPS" \
      --out "$SYNTH_DS" --split train
  # 검증용으로 71856 Validation(선박전용 2,108장)도 붙인다 — 위생 검사 전용.
  echo "prep 완료. 학습 전 datasets/ds(=71856 val) 가 있는지 확인할 것."
  ;;

stage1)
  # COCO 사전학습(yolo11s.pt)에서 출발 -> 합성 야간/IR. 실데이터는 아직 안 본다.
  yolo detect train \
    model=yolo11s.pt \
    data=configs/synth_train.yaml \
    imgsz="$IMGSZ" epochs=40 batch="$BATCH" \
    device="$DEVICE" workers="$WORKERS" \
    project=runs name="w8_stage1_synth" \
    patience=15 cache=False
  ;;

stage2)
  # stage1 가중치에서 실데이터로 파인튜닝. 최종 가중치를 실 도메인에 정렬시키는 단계.
  # lr0을 낮춰 stage1이 익힌 야간 특징을 지우지 않게 한다.
  yolo detect train \
    model=runs/w8_stage1_synth/weights/best.pt \
    data="$REAL_DATA" \
    imgsz="$IMGSZ" epochs=100 batch="$BATCH" \
    device="$DEVICE" workers="$WORKERS" \
    lr0=0.002 \
    project=runs name="w8_stage2_real" \
    patience=30 cache=False
  ;;

eval)
  W=runs/w8_stage2_real/weights/best.pt
  echo "=== [주 판정] W5 실데이터 홀드아웃 — 주간 성능 회귀 확인 ==="
  CUDA_VISIBLE_DEVICES=0 yolo detect val model="$W" data="$REAL_DATA" imgsz="$IMGSZ"
  echo "=== [위생 검사] 71856 조건별 — 합성 도메인 내부 수치, 성과로 보고하지 말 것 ==="
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_synth71856.py \
      --weights "$W" --ds datasets/ds --imgsz "$IMGSZ" \
      --out results/w8_synth71856_conditions.csv
  ;;

*) echo "알 수 없는 단계: $STEP"; exit 1 ;;
esac
