# 배포용 이미지 — ultralytics jetson 컨테이너에 부족한 것만 얹는다.
#
#   docker build -t marine-detect:jetson -f deploy/jetson/Dockerfile deploy/jetson
#
# 왜 필요한가 (전부 실제로 막혔던 것):
#   1. 기본 이미지의 OpenCV 는 **opencv-python-headless** 라 GUI 가 없다(`GUI: NONE`).
#      `cv2.imshow` 가 "The function is not implemented" 로 죽어 모니터 표시가 불가능하다.
#      → opencv-python(QT5 GUI 포함)으로 교체한다.
#   2. ByteTrack 의존성 `lap` 이 없어서 매 실행마다 자동설치되고, 그 수 초가 e2e 평균을
#      오염시킨다(§15-6). 미리 넣어둔다.
#   3. 컨테이너가 --rm 이라 위 두 가지가 매번 반복된다. 이미지에 구우면 한 번으로 끝난다.
#
# 유지되는 것: GStreamer + NVIDIA 플러그인(nvv4l2decoder/nvvidconv), TensorRT, torch.
FROM ultralytics/ultralytics:latest-jetson-jetpack6

# headless 를 먼저 지우지 않으면 두 패키지가 같은 cv2 를 놓고 충돌한다.
RUN pip uninstall -y opencv-python-headless opencv-contrib-python-headless 2>/dev/null || true \
 && pip install --no-cache-dir opencv-python lap \
 && python3 -c "import cv2, lap; \
    info = cv2.getBuildInformation(); \
    gui = [l for l in info.split(chr(10)) if 'GUI:' in l]; \
    print('cv2', cv2.__version__, gui[0].strip() if gui else 'GUI 정보 없음'); \
    assert 'NONE' not in (gui[0] if gui else 'NONE'), 'GUI 없는 OpenCV — 빌드 실패'"

WORKDIR /bundle/YOLO-pipeline
