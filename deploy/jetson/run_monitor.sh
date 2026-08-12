#!/usr/bin/env bash
# 실배포 실행 — 웹캠 입력 → 탐지·추적·경보 → 보드 모니터에 직접 표시.
#
#   bash deploy/jetson/run_monitor.sh            # 배포 설정 그대로
#   FPS_ONLY=1 bash deploy/jetson/run_monitor.sh # 표시 없이 FPS 만 (비교용)
#
# 🔴 보드 앞에서 한 번만: xhost +local:docker
#    (컨테이너가 보드의 X 화면에 그리려면 접근 허가가 필요하다. 재부팅하면 다시 해야 한다.)
#
# 배포 설정 근거는 W10_주간증강_결과정리.md §8.
#   모델 best_spot(실데이터만) · 544×960 rect · TRT FP16 · conf 0.25 · N-of-M(10중 6)
#   recall 0.9804 · 오경보 2.73% · 65.3 FPS · p95 15.87ms · 경보지연 5프레임
set -u

MODEL=${MODEL:-engines/best_spot_544x960_fp16.engine}
IMGSZ=${IMGSZ:-544,960}
PRE=${PRE:-960x540}          # 🔴 모델 입력이 아니라 '레터박스 직전' 크기 (1920×1080의 0.5배)
CONF=${CONF:-0.25}           # 확정 운용점 (conf 스윕 결과, W10 §8)
ALERT_CONF=${ALERT_CONF:-0.25}   # 경보 임계도 같은 값 — 측정한 recall 과 경보 동작을 일치시킨다
N=${N:-6}; M=${M:-10}        # N-of-M: 최근 10프레임 중 6프레임 이상 → 경보 ON
OUTDIR=${OUTDIR:-runs/deploy_live}
IMG=ultralytics/ultralytics:latest-jetson-jetpack6

cd "$HOME/bundle/YOLO-pipeline" || { echo "repo 없음"; exit 1; }

# ── 사전 점검 ────────────────────────────────────────────────────────
fail=0
echo "=== 사전 점검 ==="

G=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq 2>/dev/null || echo 0)
GM=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/max_freq 2>/dev/null || echo 1)
if [ "$G" = "$GM" ]; then echo "  클럭    OK ($((G/1000000)) MHz 고정)"
else echo "  클럭    ⚠️  $((G/1000000))/$((GM/1000000)) MHz — 'sudo jetson_clocks' 권장(§15-20)"; fi

P=$(nvpmodel -q 2>/dev/null | head -1)
echo "  전력    ${P:-조회 실패}"

if [ -e /dev/video0 ]; then echo "  카메라  OK (/dev/video0)"
else echo "  카메라  ❌ /dev/video0 없음 — USB 연결 확인"; fail=1; fi

if [ -f "$MODEL" ]; then echo "  엔진    OK ($(basename $MODEL))"
else echo "  엔진    ❌ $MODEL 없음 — deploy/export_trt.py 로 빌드"; fail=1; fi

if [ -z "${FPS_ONLY:-}" ]; then
  if [ -S /tmp/.X11-unix/X0 ]; then echo "  디스플레이 OK (:0)"
  else echo "  디스플레이 ❌ /tmp/.X11-unix/X0 없음 — 모니터가 연결된 세션에서 실행할 것"; fail=1; fi
fi

[ "$fail" = 0 ] || { echo "점검 실패 — 중단"; exit 1; }

# ── 실행 ─────────────────────────────────────────────────────────────
COMMON="--source usb:0 --gst --cam-fmt mjpg --pre-size $PRE --live-direct \
  --model $MODEL --imgsz $IMGSZ --conf $CONF --alert-conf $ALERT_CONF \
  --n $N --m $M --outdir $OUTDIR"

if [ -n "${FPS_ONLY:-}" ]; then
  echo -e "\n=== 표시 없이 실행 (FPS 기준선) ===\n"
  exec docker run --rm --runtime nvidia --shm-size=2g --network host \
    --device /dev/video0 -v "$HOME/bundle:/bundle" -w /bundle/YOLO-pipeline "$IMG" \
    bash -c "pip install -q lap >/dev/null 2>&1; python3 deploy/stream_infer.py $COMMON --limit 300 --no-snapshots"
fi

echo -e "\n=== 모니터 표시로 실행 (창에서 q 로 종료) ===\n"
exec docker run --rm -it --runtime nvidia --shm-size=2g --network host \
  --device /dev/video0 \
  -e DISPLAY="${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOME/bundle:/bundle" -w /bundle/YOLO-pipeline "$IMG" \
  bash -c "pip install -q lap >/dev/null 2>&1; python3 deploy/stream_infer.py $COMMON --show"
