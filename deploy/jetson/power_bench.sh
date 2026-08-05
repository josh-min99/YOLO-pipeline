#!/usr/bin/env bash
# 전력 모드별 × 지속 성능 벤치 (P3). 배포 숫자는 '피크'가 아니라 '열스로틀 후 지속' 값이다.
# 사용: bash deploy/jetson/power_bench.sh <engine> <imgsz> <frames_dir> [modes...]
#   예: bash deploy/jetson/power_bench.sh engines/best_spot_736x1280_fp16.engine 736x1280 frames 2 1 0
#
# 🔴 모드 번호는 보드마다 다르다. Orin Nano Super(JP6.2.1) 실측:
#      0=15W  1=25W(기본)  2=MAXN_SUPER  3=7W
#    `sudo nvpmodel -p --verbose` 로 먼저 확인할 것. 구형 문서의 `0=MAXN` 은 여기서 15W다.
set -e
ENGINE=${1:?engine 경로}
IMGSZ=${2:-736x1280}
FRAMES=${3:?프레임 폴더 또는 비디오}
shift 3 || true
MODES=${@:-0}
OUT=results/deploy
mkdir -p "$OUT"

for M in $MODES; do
  echo "=== nvpmodel -m $M ==="
  sudo nvpmodel -m "$M"
  sudo jetson_clocks || true
  sleep 5

  LOG="$OUT/tegrastats_m${M}.log"
  sudo tegrastats --interval 1000 > "$LOG" 2>&1 &
  TS=$!

  echo "--- (a) 냉간 벤치 ---"
  python3 deploy/bench_engine.py --weights "$ENGINE" --imgsz-list "$IMGSZ" \
      --frames "$FRAMES" --n 200 --out "$OUT" || true

  echo "--- (b) 10분 부하 후 지속 성능 ---"
  timeout 600 python3 deploy/stream_infer.py --source "$FRAMES" --model "$ENGINE" \
      --imgsz "${IMGSZ/x/,}" --outdir "$OUT" --tag "soak_m${M}" || true
  python3 deploy/bench_engine.py --weights "$ENGINE" --imgsz-list "$IMGSZ" \
      --frames "$FRAMES" --n 200 --out "$OUT" || true

  sudo kill $TS 2>/dev/null || true
  echo "전력/온도 로그: $LOG  (POM_5V_IN / GPU@ 온도 열 확인)"
done

cat <<'EOF'

정리할 표 (W6 §P3):
  전력모드 | 냉간 FPS | 10분 후 FPS | 평균 전력(W) | GPU 온도(°C) | 스로틀 여부
  → '10분 후 FPS'가 배포 스펙. 냉간 값만 보고하면 현장에서 안 나온다.
EOF
