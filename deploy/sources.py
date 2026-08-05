"""
입력원 추상화 — 파일 / 프레임폴더 / USB / CSI / RTSP 를 하나의 인터페이스로.

stream_infer.py 는 이 모듈만 보고, 입력원이 바뀌어도 추론 루프는 그대로 둔다.

  spec 형식
    <경로.mp4|.avi>            파일 (재현·발표용 기본값)
    dir:<폴더>[:<클립접두>]     프레임 jpg 폴더 (우리 데이터셋이 이 형태)
    usb:0                      USB 웹캠 (V4L2)
    csi:0                      Jetson CSI 카메라 (nvarguscamerasrc)
    rtsp://user:pw@host/path   IP카메라/CCTV

  🔴 라이브 입력(usb/csi/rtsp)의 핵심 규칙: **처리보다 입력이 빠르면 최신 프레임만 남기고 버린다.**
     버퍼링하면 지연이 계속 누적돼서 "실시간"이 아니게 됨. 백그라운드 스레드가 계속 읽어
     최신 1장만 들고 있고, 버린 장수를 dropped 로 센다(측정·보고용).
     파일/폴더는 반대로 한 장도 버리지 않는다(정량 평가는 전 프레임을 봐야 하므로).
"""
import threading
import time
from pathlib import Path

import cv2


class FrameSource:
    """공통 인터페이스: for idx, ts, frame in source: ...  + source.stats"""

    kind = "base"
    is_live = False

    def __iter__(self):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def stats(self):
        return dict(kind=self.kind, read=getattr(self, "_read", 0),
                    dropped=getattr(self, "_dropped", 0))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class DirSource(FrameSource):
    """프레임 jpg 폴더. datasets/marine_frames 처럼 flat 하게 깔린 프레임을 클립 단위로 읽음."""

    kind = "dir"

    def __init__(self, folder, prefix="", fps=8.0, limit=0):
        self.files = sorted(Path(folder).glob(f"{prefix}*.jpg"))
        if not self.files:
            raise SystemExit(f"프레임 없음: {folder} (prefix={prefix!r})")
        if limit:
            self.files = self.files[:limit]
        self.dt = 1.0 / fps if fps > 0 else 0.0
        self._read = 0
        self._dropped = 0

    def __iter__(self):
        for i, f in enumerate(self.files):
            # 한글 경로 대비: imread 대신 imdecode (유의사항 §9-4)
            import numpy as np
            frame = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            self._read += 1
            yield i, i * self.dt, frame


class FileSource(FrameSource):
    """비디오 파일. 프레임을 버리지 않음(정량 비교용)."""

    kind = "file"

    def __init__(self, path, limit=0, loop=False):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"파일 열기 실패: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.limit, self.loop = limit, loop
        self._read = 0
        self._dropped = 0

    def __iter__(self):
        i = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            self._read += 1
            yield i, i / self.fps, frame
            i += 1
            if self.limit and i >= self.limit:
                break

    def close(self):
        self.cap.release()


class LiveSource(FrameSource):
    """usb / csi / rtsp 공통 — 백그라운드 리더 + 최신 프레임만 유지(오래된 건 drop)."""

    is_live = True

    def __init__(self, cap, kind, limit=0, timeout=10.0):
        self.cap, self.kind = cap, kind
        if not self.cap.isOpened():
            raise SystemExit(f"{kind} 입력 열기 실패 — 파이프라인/URL/권한 확인")
        self.limit, self.timeout = limit, timeout
        self._latest = None          # (grab_ts, frame)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._read = 0               # 실제로 추론에 넘어간 장수
        self._grabbed = 0            # 카메라에서 읽은 장수
        self._dropped = 0            # 읽었지만 안 쓰고 버린 장수
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            self._grabbed += 1
            with self._lock:
                if self._latest is not None:
                    self._dropped += 1   # 이전 프레임을 못 쓰고 버림
                self._latest = (time.time(), frame)

    def __iter__(self):
        i = 0
        t_wait = time.time()
        while True:
            with self._lock:
                item, self._latest = self._latest, None
            if item is None:
                if time.time() - t_wait > self.timeout:
                    print(f"[{self.kind}] {self.timeout}s 동안 프레임 없음 — 종료")
                    break
                time.sleep(0.001)
                continue
            t_wait = time.time()
            ts, frame = item
            self._read += 1
            yield i, ts, frame
            i += 1
            if self.limit and i >= self.limit:
                break

    def close(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        self.cap.release()

    @property
    def stats(self):
        s = super().stats
        s.update(grabbed=self._grabbed,
                 drop_rate=self._dropped / max(1, self._grabbed))
        return s


def _scaled_caps(out_w, out_h):
    """nvvidconv 출력 caps. out_* 를 주면 VIC 가 리사이즈까지 해서 CPU 전처리를 걷어낸다.

    🔴 리사이즈 목표는 '모델 입력 크기'가 아니라 '레터박스 직전 크기'다.
       예) 1920×1080 → 모델 960×544 인 경우 목표는 **960×540**.
       960×544 로 직접 늘리면 세로가 0.74% 늘어나(aspect 왜곡) 학습 때 본 것과 다른 픽셀이 된다.
       960×540 으로 주면 ultralytics LetterBox 가 r=1.0 으로 판단해 리사이즈를 건너뛰고
       패딩 4줄만 붙인다 — 이게 우리가 원하는 상태다.
    """
    if out_w and out_h:
        return f"video/x-raw,format=BGRx,width={out_w},height={out_h}"
    return "video/x-raw,format=BGRx"


def _usb_pipeline(idx, width, height, fps, fmt="mjpg", out_w=None, out_h=None):
    """USB(UVC) 웹캠을 HW 경로로. MJPG 는 nvv4l2decoder(전용 디코더)로 푼다.

    fmt='mjpg'  : 카메라가 JPEG 로 보내고 보드가 HW 디코딩 (1080p30 가능, 압축 있음)
    fmt='yuyv'  : 무압축. 디코딩 불필요하나 USB 대역폭 때문에 1080p 는 보통 5fps
    """
    src = f"v4l2src device=/dev/video{idx} io-mode=2"
    if fmt == "mjpg":
        head = (f"{src} ! image/jpeg,width={width},height={height},framerate={fps}/1 ! "
                f"nvv4l2decoder mjpeg=1 ! ")
    else:
        head = (f"{src} ! video/x-raw,format=YUY2,width={width},height={height},"
                f"framerate={fps}/1 ! ")
    return (head + f"nvvidconv ! {_scaled_caps(out_w, out_h)} ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")


def _csi_pipeline(sensor_id, width, height, fps):
    # Jetson 전용. ISP·리사이즈를 HW(nvarguscamerasrc/nvvidconv)에 맡김.
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")


def _rtsp_pipeline(url, latency=100):
    # 🔴 디코딩을 CPU로 하면 그게 병목이 된다 → Jetson HW 디코더(nvv4l2decoder) 사용.
    return (f"rtspsrc location={url} latency={latency} ! rtph264depay ! h264parse ! "
            f"nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")


def open_source(spec, fps=8.0, limit=0, loop=False, width=1920, height=1080,
                cam_fps=30, gst=None, pre_w=None, pre_h=None, cam_fmt="mjpg"):
    """spec 문자열 → FrameSource.

    gst   : None 이면 입력원별 기본값(csi/rtsp는 True, usb는 False=기존 CPU 경로)
    pre_w/pre_h : 주면 nvvidconv 가 그 크기로 리사이즈(=CPU 전처리 오프로딩). gst 경로에서만 유효
    cam_fmt : usb 전용. 'mjpg'(HW 디코딩) 또는 'yuyv'(무압축)
    """
    spec = str(spec)

    if spec.startswith("dir:"):
        parts = spec.split(":", 2)
        folder = parts[1]
        prefix = parts[2] if len(parts) > 2 else ""
        return DirSource(folder, prefix, fps=fps, limit=limit)

    if spec.startswith("usb:"):
        idx = int(spec.split(":", 1)[1])
        if gst:
            pipe = _usb_pipeline(idx, width, height, cam_fps, cam_fmt, pre_w, pre_h)
            print(f"[usb] gstreamer: {pipe}")
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return LiveSource(cap, "usb-gst", limit)
            print("[usb] gstreamer 파이프라인 실패 → V4L2 폴백(CPU 디코딩·리사이즈)")
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY)
        # 🔴 FOURCC 를 먼저, 그리고 반드시 지정한다.
        #    지정하지 않으면 V4L2 가 기본 포맷 YUYV(무압축)를 고르는데, 1080p 는 프레임당 ~4MB라
        #    USB 대역폭에 막혀 **5 FPS**밖에 안 나온다. 카메라 성능이 아니라 포맷 선택 문제다.
        #    (C920 실측: YUYV 1080p ≈ 5 FPS / MJPG 1080p ≈ 29.9 FPS)
        fourcc = "MJPG" if cam_fmt == "mjpg" else "YUYV"
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, cam_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 지연 누적 방지
        got = int(cap.get(cv2.CAP_PROP_FOURCC))
        print(f"[usb] fourcc={''.join(chr((got >> 8 * i) & 255) for i in range(4))} "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
              f"@{cap.get(cv2.CAP_PROP_FPS):.0f}fps (요청 {fourcc} {width}x{height}@{cam_fps})")
        return LiveSource(cap, "usb", limit)

    if spec.startswith("csi:"):
        sid = int(spec.split(":", 1)[1])
        pipe = _csi_pipeline(sid, width, height, cam_fps)
        print(f"[csi] {pipe}")
        return LiveSource(cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER), "csi", limit)

    if spec.startswith("rtsp://"):
        use_gst = _gst_available() if gst is None else gst
        if use_gst:
            pipe = _rtsp_pipeline(spec)
            print(f"[rtsp] gstreamer: {pipe}")
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return LiveSource(cap, "rtsp", limit)
            print("[rtsp] gstreamer 파이프라인 실패 → FFMPEG 폴백(CPU 디코딩, 느릴 수 있음)")
        cap = cv2.VideoCapture(spec, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return LiveSource(cap, "rtsp", limit)

    if spec.startswith("file:"):
        spec = spec.split(":", 1)[1]
    return FileSource(spec, limit=limit, loop=loop)


def _gst_available():
    try:
        return "GStreamer:                   YES" in cv2.getBuildInformation() or \
               "GStreamer" in cv2.getBuildInformation().split("Video I/O")[1].split("Parallel")[0]
    except Exception:
        return False


if __name__ == "__main__":   # 빠른 확인: python deploy/sources.py <spec>
    import sys
    src = open_source(sys.argv[1] if len(sys.argv) > 1 else "usb:0", limit=30)
    t0 = time.time()
    with src:
        for i, ts, f in src:
            pass
    print(src.stats, f"{time.time()-t0:.2f}s")
