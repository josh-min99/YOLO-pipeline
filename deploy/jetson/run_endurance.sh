#!/usr/bin/env bash
# 연속 부하 시험 — 배포 형태 그대로 30분 돌리며 FPS·지연·온도·클럭 추이를 남긴다.
#
#   bash deploy/jetson/run_endurance.sh            # 30분 (기본)
#   DURATION=60 STATS=15 bash deploy/jetson/run_endurance.sh   # 짧은 리허설
#
# 🔴 배포 스펙은 냉간값이 아니라 '열이 오른 뒤에도 유지되는 값'이다.
#    그래서 평균이 아니라 **첫 구간 대비 마지막 구간**으로 판정한다(§15-6 과 같은 이유).
# 🔴 표시(--show)를 켠 채로 잰다 — 실제 배포 형태와 다른 조건에서 재면 의미가 없다.
set -u

DURATION=${DURATION:-1800}    # 초
STATS=${STATS:-60}            # 구간 길이(초)
MODEL=${MODEL:-engines/best_spot_544x960_fp16.engine}
IMGSZ=${IMGSZ:-544,960}
PRE=${PRE:-960x540}
CONF=${CONF:-0.25}
N=${N:-6}; M=${M:-10}
OUTDIR=${OUTDIR:-runs/endurance}
IMG=${IMG:-marine-detect:jetson}
SHOW=${SHOW:---show}          # SHOW= 로 비우면 표시 없이

cd "$HOME/bundle/YOLO-pipeline" || { echo "repo 없음"; exit 1; }

G=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq 2>/dev/null || echo 0)
GM=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/max_freq 2>/dev/null || echo 1)
[ "$G" = "$GM" ] || { echo "🔴 GPU $((G/1000000))/$((GM/1000000)) MHz — sudo jetson_clocks 먼저(§15-20)"; exit 1; }
echo "클럭 $((G/1000000)) MHz 고정 · $(nvpmodel -q 2>/dev/null | head -1)"
echo "시험 $((DURATION/60))분 · 구간 ${STATS}초 · 표시 ${SHOW:-없음}"

# --rm 을 붙이지 않는다 — 중간에 끊겨도 docker logs 로 되짚을 수 있어야 한다.
docker rm -f endurance >/dev/null 2>&1
exec docker run --name endurance --runtime nvidia --shm-size=2g --network host \
  --device /dev/video0 \
  -e DISPLAY="${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOME/bundle:/bundle" -w /bundle/YOLO-pipeline "$IMG" \
  python3 deploy/stream_infer.py \
    --source usb:0 --gst --cam-fmt mjpg --pre-size "$PRE" --live-direct \
    --model "$MODEL" --imgsz "$IMGSZ" --conf "$CONF" --alert-conf "$CONF" \
    --n "$N" --m "$M" --outdir "$OUTDIR" --no-snapshots --tag endurance \
    --duration "$DURATION" --stats-every "$STATS" $SHOW
