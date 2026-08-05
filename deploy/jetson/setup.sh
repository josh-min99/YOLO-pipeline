#!/usr/bin/env bash
# Jetson 셋업 점검 — 보드에 처음 붙었을 때 이걸 먼저 돌린다(P3).
# 사용: bash deploy/jetson/setup.sh
#
# 🔴 가장 큰 함정: `pip install ultralytics` 를 그냥 하면 aarch64용 **CPU 전용 torch**가 깔린다.
#    Jetson에서 GPU torch는 JetPack 버전에 맞는 NVIDIA 휠(또는 컨테이너)로만 온다.
#    → 권장: NVIDIA/dustynv 컨테이너 사용. 맨 아래 안내 참고.
set -e

echo "=== 1. 보드/JetPack ==="
cat /etc/nv_tegra_release 2>/dev/null || echo "  (L4T 정보 없음 — Jetson이 아닐 수 있음)"
[ -f /sys/firmware/devicetree/base/model ] && tr -d '\0' < /sys/firmware/devicetree/base/model && echo

echo -e "\n=== 2. TensorRT / CUDA ==="
TRTEXEC=/usr/src/tensorrt/bin/trtexec
if [ -x "$TRTEXEC" ]; then
  echo "  trtexec: $TRTEXEC"
  dpkg -l 2>/dev/null | grep -E "tensorrt|nvinfer" | head -3 || true
else
  echo "  ⚠️ trtexec 없음 — JetPack(SDK Manager) 설치 확인"
fi
nvcc --version 2>/dev/null | tail -1 || echo "  nvcc 없음(런타임만 있을 수 있음 — 정상)"

echo -e "\n=== 3. 모니터링 도구(jtop) ==="
if ! command -v jtop >/dev/null; then
  echo "  설치: sudo pip3 install -U jetson-stats && sudo reboot"
else
  echo "  jtop OK — 벤치 중 다른 터미널에서 jtop 으로 전력/온도/클럭 확인"
fi

echo -e "\n=== 4. 전력 모드 ==="
sudo nvpmodel -q 2>/dev/null || echo "  nvpmodel 조회 실패"
cat <<'EOF'
  측정 규율(P3): 전력 모드별로 따로 잰다. 피크가 아니라 **열스로틀 후 지속 FPS**가 배포 숫자.
    sudo nvpmodel -m 0       # MAXN
    sudo jetson_clocks       # 클럭 고정(측정 분산 감소)
EOF

echo -e "\n=== 5. torch/ultralytics 상태 ==="
python3 - <<'PY' 2>/dev/null || echo "  python3 환경에 torch 없음"
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  [!] CPU 전용 torch가 깔려 있음 — Jetson용 휠/컨테이너로 교체 필요")
PY

cat <<'EOF'

=== 권장 실행 방식 (의존성 지옥 회피) ===
JetPack 6.x 기준, GPU torch+ultralytics+TRT가 이미 맞춰진 컨테이너를 쓰는 게 가장 빠르다:

  sudo docker run -it --rm --runtime nvidia --network host \
    -v $(pwd):/work -w /work \
    ultralytics/ultralytics:latest-jetson-jetpack6

컨테이너 안에서 (🔴 .onnx 가 아니라 **.pt** 에서 빌드한다 — ONNX는 ultralytics export가 거부한다):
  python3 deploy/export_trt.py --best best_spot.pt --imgsz 736,1280 --formats fp16   # 엔진은 여기서 빌드
  python3 deploy/bench_engine.py --weights engines/best_spot_736x1280_fp16.engine --imgsz-list 736x1280 --frames <프레임폴더>
  python3 deploy/stream_infer.py --source rtsp://... --model engines/best_spot_736x1280_fp16.engine

ONNX밖에 없을 때만(폴백) — 클래스명 메타데이터 없는 엔진이 나온다:
  /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best_fp16.engine --fp16 \
      --shapes=images:1x3x736x1280 --memPoolSize=workspace:2048
  (export_trt.py 에 .onnx 를 주면 위 명령을 대신 실행해준다)
EOF

echo -e "\n=== 6. 비디오 인코더/디코더 (오버레이 저장 비용 판단용) ==="
if command -v gst-inspect-1.0 >/dev/null; then
  gst-inspect-1.0 2>/dev/null | grep -E "nvv4l2decoder|nvv4l2h264enc|nvvidconv" || \
    echo "  nvv4l2h264enc 없음 = HW 인코더 없음(Orin Nano) → mp4 저장은 CPU. FPS 측정 시 --save 끄기"
else
  echo "  gst-inspect-1.0 없음 — HW 인코더 유무는 mp4 저장 시 CPU 사용률로 판단"
fi
echo -e "\n점검 끝."
