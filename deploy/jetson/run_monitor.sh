#!/usr/bin/env bash
# 실배포 실행 — 웹캠 입력 → 탐지·추적·경보 → 보드 모니터에 직접 표시.
#
#   bash deploy/jetson/run_monitor.sh            # 배포 설정 그대로
#   FPS_ONLY=1 bash deploy/jetson/run_monitor.sh # 표시 없이 FPS 만 (비교용)
#   REC=1 bash deploy/jetson/run_monitor.sh      # 영상까지 기록 (현장 채증용)
#
# 사전 준비는 스크립트가 알아서 한다 — 클럭 고정(jetson_clocks), 화면 권한(xhost),
# 카메라·엔진·이미지 확인. 하나라도 어긋나면 실행하지 않고 무엇을 해야 하는지 알려준다.
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
# 배포 이미지(GUI OpenCV + lap 이 구워져 있음). 없으면 기본 이미지로 폴백하되
# 그 경우 --show 는 동작하지 않는다(headless OpenCV).
IMG=${IMG:-marine-detect:jetson}

cd "$HOME/bundle/YOLO-pipeline" || { echo "repo 없음"; exit 1; }

# ── 사전 점검 ────────────────────────────────────────────────────────
fail=0
echo "=== 사전 점검 ==="

G=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq 2>/dev/null || echo 0)
GM=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/max_freq 2>/dev/null || echo 1)
if [ "$G" = "$GM" ]; then echo "  클럭    OK ($((G/1000000)) MHz 고정)"
else
  # 🔴 경고로 끝내지 않는다. 안 걸린 채로 돌면 조용히 절반 속도(26 FPS)로 동작하고,
  #    현장에서는 그걸 알아챌 방법이 없다(§15-20). 재부팅하면 매번 풀린다.
  echo "  클럭    ⚠️  $((G/1000000))/$((GM/1000000)) MHz — 최대가 아니다. 고정 시도:"
  echo "          (비밀번호를 물으면 입력할 것. ssh 로는 실패할 수 있다)"
  # 🔴 stderr 를 버리지 않는다 — sudo 의 비밀번호 프롬프트가 stderr 로 나가므로,
  #    묻어버리면 화면엔 아무것도 없는데 입력을 기다리는 '멈춘 것처럼 보이는' 상태가 된다.
  sudo jetson_clocks
  G=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq 2>/dev/null || echo 0)
  if [ "$G" = "$GM" ]; then echo "          → OK ($((G/1000000)) MHz 고정됨)"
  else echo "          → ❌ 실패. 터미널에서 'sudo jetson_clocks' 실행 후 다시 시작할 것"; fail=1; fi
fi

P=$(nvpmodel -q 2>/dev/null | head -1)
echo "  전력    ${P:-조회 실패}"

if [ -e /dev/video0 ]; then echo "  카메라  OK (/dev/video0)"
else echo "  카메라  ❌ /dev/video0 없음 — USB 연결 확인"; fail=1; fi

if [ -f "$MODEL" ]; then echo "  엔진    OK ($(basename $MODEL))"
else echo "  엔진    ❌ $MODEL 없음 — deploy/export_trt.py 로 빌드"; fail=1; fi

if [ -z "${FPS_ONLY:-}" ]; then
  if [ -S /tmp/.X11-unix/X0 ]; then echo "  디스플레이 OK (:0)"
  else echo "  디스플레이 ❌ /tmp/.X11-unix/X0 없음 — 모니터가 연결된 세션에서 실행할 것"; fail=1; fi
  # 컨테이너가 보드 화면에 그릴 권한. 재부팅하면 풀리므로 매번 걸어둔다(로컬 접속만 허용).
  if xhost 2>/dev/null | grep -q "^LOCAL:"; then echo "  화면권한 OK (LOCAL 허용됨)"
  else
    if xhost +local:docker >/dev/null 2>&1; then echo "  화면권한 OK (방금 허용함)"
    else echo "  화면권한 ⚠️  xhost 실패 — 원격(ssh)이면 정상. 보드 터미널에서 'xhost +local:docker'"; fi
  fi
fi

if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "  이미지  ❌ $IMG 없음 — 먼저 빌드:"
  echo "          docker build -t marine-detect:jetson -f deploy/jetson/Dockerfile deploy/jetson"
  fail=1
else
  echo "  이미지  OK ($IMG)"
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
    python3 deploy/stream_infer.py $COMMON --limit 300 --no-snapshots
fi

# 현장 기록 — REC=1 로 켠다.
# 🔴 Orin Nano 에는 NVENC 가 없어 인코딩이 전부 CPU 다(§15-16). 비용은 **미측정** —
#    현장에서 켤 거면 FPS 표시를 보면서 판단할 것. 경보 스냅샷은 켜든 끄든 남는다.
SAVE=""
if [ -n "${REC:-}" ]; then
  SAVE="--save $OUTDIR/rec_$(date +%m%d_%H%M).mp4"
  echo "🔴 영상 기록 켬 — CPU 인코딩이라 FPS 가 떨어질 수 있다(미측정)"
fi

echo -e "\n=== 모니터 표시로 실행 (창에서 q 로 종료) ===\n"
echo "  이벤트 로그 : ~/bundle/YOLO-pipeline/$OUTDIR/events.jsonl"
echo "  경보 스냅샷 : ~/bundle/YOLO-pipeline/$OUTDIR/  (경보 발생 시 자동 저장)"
echo
# -t 는 붙이지 않는다 — nohup/백그라운드로 띄우면 TTY 가 없어
# "the input device is not a TTY" 로 죽는다. 표시에는 TTY 가 필요 없다(X11 소켓만 있으면 된다).
# 창 종료는 q 키, 원격 종료는 docker stop.
exec docker run --rm --runtime nvidia --shm-size=2g --network host \
  --device /dev/video0 \
  -e DISPLAY="${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOME/bundle:/bundle" -w /bundle/YOLO-pipeline "$IMG" \
  python3 deploy/stream_infer.py $COMMON --show --stats-every 60 $SAVE
