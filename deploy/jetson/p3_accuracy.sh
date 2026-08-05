#!/usr/bin/env bash
# P3 정확도 패리티 — 같은 보드에서 PyTorch(.pt) vs TensorRT(FP16) 를 나란히 잰다.
#
#   bash deploy/jetson/p3_accuracy.sh          # 4개 조합 전부, 60~90분
#   LIMIT=500 bash deploy/jetson/p3_accuracy.sh  # 리허설(500장) — 기준값과 비교 불가
#
# 왜 보드에서 .pt 도 재는가:
#   "TensorRT가 정확도를 깎았나"는 **같은 기계에서 나란히 재야** 답이 된다.
#   x86 숫자와 비교하면 하드웨어·TRT버전·프레임워크가 한꺼번에 바뀌어 원인 분리가 안 된다.
#
# 축이 두 개인 이유(§15-6): ultralytics val() 은 rect(imgsz=[h,w])를 못 받는다.
#   · mAP   축 = 정사각 1280  (DATASET.md §6-2 벤치마크 규약, 기준 0.9788)
#   · 운영점 축 = rect 736x1280 (실제 배포 설정, 기준 recall@0.6 0.9755 / 오경보 2.12%)
# 🔴 컨테이너에서 돌릴 것: docker run --shm-size=2g -v $HOME/bundle:/bundle ...
#    --shm-size 를 빼면 도커 기본 /dev/shm 64MB 로는 PyTorch 데이터로더가 죽는다.
set -u
W=/bundle/weights/best_spot.pt
ROOT=/bundle/benchmark/marine_session_spot
YAML=/bundle/marine_board.yaml
OUT=results/deploy
LIMIT=${LIMIT:-0}
LOG=$OUT/acc_$(date +%m%d_%H%M).log

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

step() { echo -e "\n\n########## $* ##########\n"; }
echo "정확도 패리티 시작 $(date '+%F %T')  LIMIT=$LIMIT"
[ "$LIMIT" != "0" ] && echo "⚠️ LIMIT 사용 = 리허설이다. 기준값(0.9788/0.9755)과 비교하지 말 것."

pip install -q lap 2>&1 | tail -1 || true

step "0. 정사각 1280 엔진 (mAP 축용)"
if [ -f engines/best_spot_1280_fp16.engine ]; then
  echo "  이미 있음, 건너뜀"
else
  python3 deploy/export_trt.py --best "$W" --imgsz 1280 --formats fp16 --workspace 2
fi

# ── mAP 축 (정사각 1280, conf 0.001 기본) ─────────────────────────────
step "1. mAP — PyTorch .pt @1280   (x86 PyTorch 기준: 군함 0.9788 / 전체 0.9613)"
yolo detect val model="$W" data="$YAML" imgsz=1280 batch=1 plots=False

step "2. mAP — TensorRT FP16 @1280 (x86 TRT 기준: 군함 0.9789 = 손실 0)"
yolo detect val model=engines/best_spot_1280_fp16.engine data="$YAML" imgsz=1280 batch=1 plots=False

# ── 운영점 축 (rect 736x1280, conf 0.6) ───────────────────────────────
LIM_ARG=""
[ "$LIMIT" != "0" ] && LIM_ARG="--limit $LIMIT"

step "3. 운영점 — PyTorch .pt @736x1280 conf0.6"
python3 deploy/eval_op.py --weights "$W" --imgsz 736x1280 --data-root "$ROOT" \
    --conf 0.6 --csv "$OUT/op_parity.csv" $LIM_ARG

step "4. 운영점 — TensorRT FP16 @736x1280 conf0.6  (x86 기준: recall 0.9755 / 오경보 2.12%)"
python3 deploy/eval_op.py --weights engines/best_spot_736x1280_fp16.engine --imgsz 736x1280 \
    --data-root "$ROOT" --conf 0.6 --csv "$OUT/op_parity.csv" $LIM_ARG

step "끝 $(date '+%F %T')"
echo "운영점 요약:"; cat "$OUT/op_parity.csv"
cat <<EOF

판정 기준:
  · FP16 변환 손실은 **1%p 이하**여야 한다. 그보다 크면 TRT 10.7 쪽을 의심할 근거가 된다.
  · 같은 보드에서 .pt 와 engine 을 나란히 쟀으므로, 차이가 있으면 그건 순수하게 엔진 탓이다.
  · 로그 전체: $LOG
EOF
