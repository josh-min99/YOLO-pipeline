"""
배포 실행 엔트리 — 입력원 → 추론 → 추적 → 경보 → 출력. 이 파일 하나가 '제품'이다.

  # 파일/프레임폴더 (재현·발표용)
  python deploy/stream_infer.py --source demo.mp4 --model best.pt --save out.mp4
  python deploy/stream_infer.py --source dir:datasets/marine_frames:I2_S0_C5_0079 --model best.pt

  # 라이브 (보드에서)
  python deploy/stream_infer.py --source rtsp://... --model best.engine --imgsz 736,1280
  python deploy/stream_infer.py --source usb:0 --model best.engine --show

  # ultralytics 없이 배관만 점검(로컬 CPU에서 가능)
  python deploy/stream_infer.py --source demo.mp4 --stub --limit 60

가중치는 .pt / .engine(TensorRT) / .onnx 아무거나. 같은 코드로 돌아야 P1~P3 비교가 성립한다.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from alert import AlertConfig, AlertEngine, EventLogger
from sources import open_source

COLORS = {0: (0, 200, 0), 1: (0, 170, 255), 2: (0, 0, 255)}   # BGR: 어선/상선/군함


def parse_imgsz(s):
    """'1280' → 1280,  '736,1280' → [736, 1280] (rect: 레터박스 패딩 제거)"""
    if "," in str(s):
        h, w = [int(v) for v in str(s).split(",")]
        return [h, w]
    return int(s)


class Detector:
    """ultralytics 래퍼. .pt/.engine/.onnx 동일 인터페이스."""

    def __init__(self, weights, imgsz=1280, conf=0.25, track=True, device=0, half=False,
                 classes=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.track = imgsz, conf, track
        self.kw = dict(imgsz=imgsz, conf=conf, device=device, verbose=False, classes=classes)
        if half:      # ultralytics 8.4에서 half는 deprecated — 필요할 때만 전달
            self.kw["half"] = True
        self.names = self.model.names

    def infer(self, frame):
        if self.track:
            res = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **self.kw)[0]
        else:
            res = self.model.predict(frame, **self.kw)[0]
        dets = []
        for b in res.boxes:
            dets.append(dict(cls=int(b.cls.item()), conf=float(b.conf.item()),
                             xyxy=[float(v) for v in b.xyxy[0].tolist()],
                             track_id=int(b.id.item()) if b.id is not None else None))
        return dets, dict(res.speed)   # {'preprocess','inference','postprocess'} ms


class StubDetector:
    """ultralytics 없이 배관(입력·경보·오버레이·기록)만 점검하기 위한 가짜 탐지기."""

    names = {0: "fishing_boat", 1: "merchant_ship", 2: "warship"}

    def __init__(self, **_):
        self.i = -1

    def infer(self, frame):
        self.i += 1
        h, w = frame.shape[:2]
        dets = []
        if 10 <= self.i < 60:            # 진짜 표적처럼 연속 등장
            x = 0.3 * w + self.i * 2
            dets.append(dict(cls=2, conf=0.88, xyxy=[x, h * 0.45, x + 90, h * 0.52], track_id=1))
        if self.i in (5, 70):            # 단발 오탐 — 경보가 되면 안 됨
            dets.append(dict(cls=2, conf=0.72, xyxy=[10, 10, 60, 40], track_id=99))
        time.sleep(0.002)
        return dets, dict(preprocess=0.0, inference=2.0, postprocess=0.0)


def draw(frame, dets, alert_ids, names, alarm, fps_txt=""):
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        c = d["cls"]
        col = COLORS.get(c, (255, 255, 255))
        on = d.get("track_id") in alert_ids and c == 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if on else 2)
        tid = f"id:{d['track_id']} " if d.get("track_id") is not None else ""
        label = f"{tid}{names.get(c, c)} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
    if alarm:
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6)
        cv2.putText(frame, "WARSHIP ALERT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 0, 255), 4)
    if fps_txt:
        cv2.putText(frame, fps_txt, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="파일 | dir:<폴더>[:<접두>] | usb:0 | csi:0 | rtsp://...")
    ap.add_argument("--model", help="best.pt / best.engine / best.onnx")
    ap.add_argument("--stub", action="store_true", help="ultralytics 없이 배관만 점검")
    ap.add_argument("--imgsz", default="1280", help="1280 또는 736,1280(rect)")
    ap.add_argument("--conf", type=float, default=0.25, help="탐지 임계(경보 임계는 --alert-conf)")
    ap.add_argument("--alert-conf", type=float, default=0.6, help="운영점(W5 F1최적 0.604)")
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--device", default=0)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--n", type=int, default=6, help="N-of-M 경보 규칙의 N")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--off-patience", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0, help="처리 프레임 수 제한(0=전부)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="통계에서 제외할 초기 프레임 수(TRT 컨텍스트·의존성 자동설치로 첫 프레임이 수초 걸림)")
    ap.add_argument("--src-fps", type=float, default=8.0, help="dir: 소스의 가상 fps")
    ap.add_argument("--gst", action="store_true",
                    help="라이브 입력을 GStreamer HW 경로로 (nvv4l2decoder + nvvidconv)")
    ap.add_argument("--pre-size", default="",
                    help="nvvidconv 가 리사이즈할 목표 'WxH' (예: 960x540). "
                         "= CPU 전처리 오프로딩. 🔴 모델 입력이 아니라 **레터박스 직전 크기**를 준다")
    ap.add_argument("--cam-fmt", default="mjpg", choices=("mjpg", "yuyv"),
                    help="usb 전용. mjpg=HW 디코딩(압축) / yuyv=무압축")
    ap.add_argument("--cam-fps", type=int, default=30, help="카메라 요청 fps")
    ap.add_argument("--cam-size", default="1920x1080", help="카메라 요청 해상도 'WxH'")
    ap.add_argument("--live-direct", action="store_true",
                    help="라이브 입력을 직접 읽는다(프레임 안 버림). 처리가 입력보다 빠를 때 쓴다")
    ap.add_argument("--save", default="", help="오버레이 영상 저장 경로(.mp4)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--web", type=int, default=0,
                    help="웹 모니터링 뷰 포트(예: 8080). 브라우저에서 http://<보드IP>:포트")
    ap.add_argument("--web-quality", type=int, default=70, help="웹 뷰 JPEG 품질")
    ap.add_argument("--web-width", type=int, default=1280, help="웹 뷰 최대 폭(전송량 절감)")
    ap.add_argument("--outdir", default="runs/deploy", help="이벤트·스냅샷·리포트")
    ap.add_argument("--no-snapshots", action="store_true")
    ap.add_argument("--tag", default="", help="리포트 파일 이름에 붙일 식별자")
    args = ap.parse_args()

    if not args.stub and not args.model:
        raise SystemExit("--model 또는 --stub 중 하나는 필요")

    imgsz = parse_imgsz(args.imgsz)
    det = (StubDetector() if args.stub else
           Detector(args.model, imgsz=imgsz, conf=args.conf, track=not args.no_track,
                    device=args.device, half=args.half))
    names = {int(k): v for k, v in det.names.items()}

    engine = AlertEngine(AlertConfig(conf=args.alert_conf, n=args.n, m=args.m,
                                     off_patience=args.off_patience))
    outdir = Path(args.outdir)
    logger = EventLogger(outdir, save_snapshots=not args.no_snapshots)
    writer = None
    web = None
    if args.web:
        from webview import WebView
        web = WebView(args.web, quality=args.web_quality, max_width=args.web_width,
                      source=f"{args.source}  ·  {Path(str(args.model)).name}  ·  imgsz={imgsz}")

    cw, ch = [int(v) for v in args.cam_size.lower().split("x")]
    pw = ph = None
    if args.pre_size:
        pw, ph = [int(v) for v in args.pre_size.lower().split("x")]
    src = open_source(args.source, fps=args.src_fps, limit=args.limit,
                      width=cw, height=ch, cam_fps=args.cam_fps,
                      gst=args.gst or None, pre_w=pw, pre_h=ph, cam_fmt=args.cam_fmt,
                      direct=args.live_direct)
    print(f"[source] {src.kind}  [model] {'STUB' if args.stub else args.model}  imgsz={imgsz}"
          + (f"  [HW 전처리] {args.pre_size}" if args.pre_size else ""))

    t_infer, t_other, lat = [], [], []
    speeds = []
    t_start = time.time()
    idx = ts = 0
    try:
        with src:
            for idx, ts, frame in src:
                t0 = time.perf_counter()
                dets, sp = det.infer(frame)
                t1 = time.perf_counter()
                ev = engine.update(idx, ts, dets)
                # 스냅샷은 박스·경보 표시가 그려진 프레임으로 남긴다(운용자가 보는 화면 = 증거)
                need_vis = bool(args.save or args.show or web
                                or (ev and not args.no_snapshots))
                vis = None
                if need_vis:
                    fps_now = idx / max(1e-6, time.time() - t_start)
                    vis = draw(frame.copy(), dets, engine.active_ids, names, engine.alarm,
                               f"{fps_now:.1f} FPS  frame {idx}")
                if web is not None:
                    # 프레임 참조만 넘긴다(인코딩은 브라우저 스레드가 함)
                    n_done = len(lat)
                    web.publish(vis,
                                fps=(n_done / max(1e-6, time.time() - t_start)),
                                p50=(float(np.percentile(lat, 50)) if n_done > 5 else 0.0),
                                frames=idx, drop=float(src.stats.get("drop_rate", 0.0)),
                                tracks=len(engine.active_ids), alarm=bool(engine.alarm),
                                alerts=engine.summary().get("alerts", 0))
                    if ev:
                        web.add_events(ev)
                if ev:
                    logger.write(ev, vis if vis is not None else frame)
                    for e in ev:
                        print(f"  [{e['type']}] f{e['frame']} track {e['track_id']} "
                              f"{e.get('conf', e.get('conf_max', ''))}")
                if vis is not None:
                    if args.save:
                        if writer is None:
                            h, w = vis.shape[:2]
                            out_fps = getattr(src, "fps", None) or args.src_fps
                            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                                     max(1.0, out_fps), (w, h))
                        writer.write(vis)
                    if args.show:
                        cv2.imshow("marine-detect", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                t2 = time.perf_counter()
                if idx >= args.warmup:      # 워밍업 프레임은 통계에서 제외(평균이 통째로 오염됨)
                    t_infer.append((t1 - t0) * 1e3)
                    t_other.append((t2 - t1) * 1e3)
                    lat.append((t2 - t0) * 1e3)
                    speeds.append(sp)
                if idx == args.warmup - 1:
                    t_start = time.time()   # FPS 계산 기준도 워밍업 직후로 리셋
    except KeyboardInterrupt:
        print("\n중단(Ctrl+C)")
    finally:
        logger.write(engine.finalize(idx, ts))
        logger.close()
        if writer:
            writer.release()
        if web is not None:
            print("[web] 서버 종료")
            web.close()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    n = len(lat)
    if n == 0:
        raise SystemExit("처리된 프레임이 0 — 입력원 확인")
    a = np.array(lat)
    rep = dict(
        source=args.source, kind=src.kind, model="STUB" if args.stub else str(args.model),
        imgsz=imgsz, frames=n, elapsed_s=round(elapsed, 2),
        e2e_fps=round(n / elapsed, 2),
        latency_ms=dict(mean=round(a.mean(), 2), p50=round(float(np.percentile(a, 50)), 2),
                        p95=round(float(np.percentile(a, 95)), 2), max=round(a.max(), 2)),
        infer_ms=round(float(np.mean(t_infer)), 2),
        post_ms=round(float(np.mean(t_other)), 2),
        model_speed_ms={k: round(float(np.mean([s[k] for s in speeds])), 2) for k in speeds[0]},
        source_stats=src.stats, alerts=engine.summary(),
    )
    tag = args.tag or ("stub" if args.stub else Path(str(args.model)).stem)
    rp = outdir / f"report_{tag}_{time.strftime('%H%M%S')}.json"
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 결과 =====")
    print(f"프레임 {n} / {elapsed:.1f}s → end-to-end {rep['e2e_fps']} FPS")
    print(f"지연 p50 {rep['latency_ms']['p50']}ms / p95 {rep['latency_ms']['p95']}ms "
          f"(추론 {rep['infer_ms']}ms + 후처리·표시 {rep['post_ms']}ms)")
    print(f"모델 내부: {rep['model_speed_ms']}")
    print(f"입력: {src.stats}")
    print(f"경보: {engine.summary()}  → {logger.path}")
    print(f"리포트: {rp}")
    if src.is_live and src.stats.get("drop_rate", 0) > 0.05:
        print("⚠️ 입력 드롭 5% 초과 — 처리 속도가 입력 속도를 못 따라감(해상도/정밀도 조정 필요)")


if __name__ == "__main__":
    main()
