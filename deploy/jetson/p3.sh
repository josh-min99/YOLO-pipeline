#!/usr/bin/env bash
# P3 온디바이스 벤치 — 컨테이너 안에서 이 한 줄만 치면 끝난다.
#
#   bash deploy/jetson/p3.sh              # 전체 (엔진 2개 빌드 + 정확도 + 속도), 30~60분
#   bash deploy/jetson/p3.sh quick        # 엔진 rect 1개 + 속도만, 10분 (배관 확인용)
#
# 전제: 번들이 /bundle 에 마운트돼 있고 (docker run -v $HOME/bundle:/bundle),
#       작업 디렉터리가 repo 루트(/bundle/YOLO-pipeline).
#
# 🔴 저장·표시는 절대 켜지 않는다 — Orin Nano는 HW 인코더가 없어서(§15-11)
#    오버레이 인코딩이 프레임당 20ms 넘게 먹고 FPS 측정을 통째로 오염시킨다.
set -u
MODE=${1:-full}
W=/bundle/weights/best_spot.pt
ROOT=/bundle/benchmark/marine_session_spot
FRAMES=$ROOT/images/val
YAML=/bundle/marine_board.yaml
OUT=results/deploy
LOG=$OUT/p3_$(date +%m%d_%H%M).log

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1        # 화면 + 파일 동시 기록(출력 공유용)

step() { echo -e "\n\n########## $* ##########\n"; }
have() { [ -f "$1" ]; }

echo "P3 시작 $(date '+%F %T')  mode=$MODE"
echo "로그: $LOG"

step "0. 환경"
python3 -c "import torch,ultralytics,tensorrt;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| ultralytics',ultralytics.__version__,'| trt',tensorrt.__version__)"
if ! python3 -c "import torch;exit(0 if torch.cuda.is_available() else 1)"; then
  echo "🔴 cuda=False — CPU torch다(§15-3). 여기서 중단. 이 컨테이너로는 측정 의미 없음."
  exit 1
fi
for p in "$W" "$FRAMES" "$YAML"; do
  [ -e "$p" ] || { echo "🔴 없음: $p  (docker -v 마운트 확인)"; exit 1; }
done
echo "번들 OK"

step "1. 엔진 빌드 — rect 736x1280 (배포 설정)"
have engines/best_spot_736x1280_fp16.engine \
  && echo "  이미 있음, 건너뜀" \
  || python3 deploy/export_trt.py --best "$W" --imgsz 736,1280 --formats fp16 --workspace 2

if [ "$MODE" != "quick" ]; then
  step "2. 엔진 빌드 — 정사각 1280 (mAP 비교 축: val()이 rect를 못 받음, §15-6)"
  have engines/best_spot_1280_fp16.engine \
    && echo "  이미 있음, 건너뜀" \
    || python3 deploy/export_trt.py --best "$W" --imgsz 1280 --formats fp16 --workspace 2

  step "3. 정확도 A — 표준 mAP (기준: 군함 mAP50 0.9788 / 전체 0.9613)"
  yolo detect val model=engines/best_spot_1280_fp16.engine data="$YAML" imgsz=1280 plots=False

  step "4. 정확도 B — 운영점 rect (기준: recall@0.6 0.9755 / 오경보 2.12%)"
  python3 deploy/eval_op.py --weights engines/best_spot_736x1280_fp16.engine \
      --imgsz 736x1280 --data-root "$ROOT" --conf 0.6 --csv "$OUT/op_jetson.csv"
fi

step "5. 속도 — 지연 p50/p95 (x86 기준: infer 2.25ms)"
python3 deploy/bench_engine.py --weights engines/best_spot_736x1280_fp16.engine \
    --imgsz-list 736x1280 --frames "$FRAMES" --n 200 --out "$OUT"

step "6. end-to-end 헤드리스 (x86 기준: p50 9.43ms / ~106 FPS)"
DEMO=$(ls /bundle/demo/* 2>/dev/null | head -1)
python3 deploy/stream_infer.py --source "$DEMO" \
    --model engines/best_spot_736x1280_fp16.engine --imgsz 736,1280 \
    --conf 0.25 --alert-conf 0.6 --n 6 --m 10 --outdir runs/deploy --no-snapshots

step "끝"
cat <<EOF
결과 위치:
  $LOG                     ← 이 파일 하나만 공유하면 전부 들어있음
  $OUT/op_jetson.csv       운영점 지표
  $OUT/bench_*.csv         지연·FPS
  runs/deploy/report_*.json  e2e

읽을 때 (§15-4, §15-6):
  · 평균 말고 **p50**. 첫 프레임이 TRT 워밍업으로 수 초 걸려 평균을 오염시킨다.
  · 이 숫자는 현재 전력 모드 기준이다. \`sudo nvpmodel -q\` 로 확인하고 표에 같이 적을 것.
  · 배포 숫자는 열스로틀 후 지속 성능 → deploy/jetson/power_bench.sh
EOF
