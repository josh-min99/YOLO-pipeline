"""비트레이트별 탐지 성능 — RTSP(H.264) 도입 전에 반드시 재야 하는 것.

  python3 deploy/eval_compression.py --weights engines/best_544x960_fp16.engine \
      --imgsz 544x960 --data-root /bundle/benchmark/marine_session_spot \
      --bitrates 8000,4000,2000,1000 --max-clips 40 --csv results/deploy/compression.csv

왜 필요한가(§15-12):
  같은 프레임을 원본 JPEG 로 넣으면 군함 conf 0.907, MJPG 재인코딩본으로 넣으면 상선 0.547 이었다.
  픽셀 평균차는 0.8%에 불과했다. 표적이 수평선의 소형 선박이라 압축 손실이 그 윤곽에 집중되기 때문이다.
  운용 입력은 RTSP = H.264 압축이므로 **카메라 비트레이트가 탐지 성능을 좌우할 수 있다.**

방법:
  val 클립을 프레임 순서대로 H.264 로 인코딩(x264enc, 지정 비트레이트) → HW 디코더로 되돌림
  → 그 프레임으로 recall/오경보율 측정. 원본(무인코딩) 행을 같은 부분집합에서 함께 재서 비교 기준으로 둔다.

🔴 부분집합으로 돌리면 절대값을 기존 기준선(0.9755 등)과 비교하지 말 것.
   이 실험이 답하는 것은 **같은 부분집합 안에서 비트레이트에 따른 상대 변화**다.
"""
import argparse
import csv
import shlex
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_imgsz(s):
    s = str(s).replace("x", ",")
    if "," in s:
        h, w = [int(v) for v in s.split(",")]
        return [h, w]
    return int(s)


def load_labels(lab, W, H, cls):
    out = []
    for ln in Path(lab).read_text().splitlines():
        p = ln.split()
        if len(p) == 5 and int(p[0]) == cls:
            xc, yc, w, h = [float(v) for v in p[1:]]
            out.append([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H])
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / ua if ua > 0 else 0


def gst(cmd):
    r = subprocess.run(["gst-launch-1.0", "-q"] + shlex.split(cmd),
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"gst 실패: {r.stderr.decode()[-400:]}")


def encode_clip(frames, W, H, kbps, out_path, fps=30):
    """원본 프레임 → H.264(지정 비트레이트) 파일. x264enc(CPU) — Orin Nano에는 HW 인코더가 없다."""
    raw = out_path.with_suffix(".raw")
    with open(raw, "wb") as f:
        for fr in frames:
            f.write(fr.tobytes())
    gst(f"filesrc location={raw} ! rawvideoparse width={W} height={H} format=bgr "
        f"framerate={fps}/1 ! videoconvert ! x264enc bitrate={kbps} speed-preset=veryfast "
        f"key-int-max={fps} ! h264parse ! matroskamux ! filesink location={out_path}")
    raw.unlink()
    return out_path.stat().st_size


def decode_clip(path, W, H, n):
    """H.264 → 원본 해상도 BGR 프레임들. 디코딩은 HW(nvv4l2decoder)."""
    p = subprocess.Popen(
        ["gst-launch-1.0", "-q"] + shlex.split(
            f"filesrc location={path} ! matroskademux ! h264parse ! nvv4l2decoder ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            f"fdsink fd=1"),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    fsize = W * H * 3
    out, buf = [], bytearray(fsize)
    for _ in range(n):
        view, got = memoryview(buf), 0
        while got < fsize:
            k = p.stdout.readinto(view[got:])
            if not k:
                break
            got += k
        if got < fsize:
            break
        out.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy())
    p.terminate()
    return out


def score(model, frames, labs, imgsz, conf, cls, acc):
    for fr, lab in zip(frames, labs):
        H, W = fr.shape[:2]
        r = model.predict(fr, imgsz=imgsz, conf=conf, classes=[cls], verbose=False)[0]
        gt = load_labels(lab, W, H, cls)
        pred = [[float(v) for v in b.xyxy[0].tolist()] for b in r.boxes]
        if gt:
            acc["n_pos"] += 1
            used = set()
            for g in gt:
                hit = -1
                for j, pr in enumerate(pred):
                    if j not in used and iou(g, pr) >= 0.5:
                        hit = j
                        break
                if hit >= 0:
                    used.add(hit)
                    acc["tp"] += 1
                else:
                    acc["fn"] += 1
        else:
            acc["n_neg"] += 1
            if pred:
                acc["fp_frames"] += 1
                acc["fp_boxes"] += len(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default="/bundle/benchmark/marine_session_spot")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", default="544x960")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cls", type=int, default=2)
    ap.add_argument("--bitrates", default="8000,4000,2000,1000", help="kbps, 콤마 구분")
    ap.add_argument("--max-clips", type=int, default=40, help="0=전부. 클립 단위로 자른다(영상이므로)")
    ap.add_argument("--csv", default="results/deploy/compression.csv")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    root = Path(a.data_root)
    img_dir, lab_dir = root / "images" / a.split, root / "labels" / a.split
    by_clip = defaultdict(list)
    for lab in sorted(lab_dir.glob("*.txt")):
        if (img_dir / f"{lab.stem}.jpg").exists():
            by_clip[lab.stem[:-3]].append(lab)
    clips = sorted(by_clip)
    if a.max_clips:
        # 군함 있는 클립과 없는 클립을 섞어서 뽑는다(둘 다 있어야 recall·오경보가 나온다)
        clips = clips[:: max(1, len(clips) // a.max_clips)][:a.max_clips]
    print(f"클립 {len(clips)}개 / 프레임 {sum(len(by_clip[c]) for c in clips)}장")

    imgsz = parse_imgsz(a.imgsz)
    model = YOLO(a.weights)
    rates = [("원본", 0)] + [(f"{int(b)}kbps", int(b)) for b in a.bitrates.split(",")]
    rows = []
    tmp = Path(tempfile.mkdtemp())

    for name, kbps in rates:
        acc = defaultdict(int)
        nbytes = 0
        t0 = time.time()
        for ci, clip in enumerate(clips):
            labs = by_clip[clip]
            frames = [cv2.imread(str(img_dir / f"{l.stem}.jpg")) for l in labs]
            frames = [f for f in frames if f is not None]
            if not frames:
                continue
            H, W = frames[0].shape[:2]
            if kbps:
                mkv = tmp / f"{clip}.mkv"
                nbytes += encode_clip(frames, W, H, kbps, mkv)
                dec = decode_clip(mkv, W, H, len(frames))
                mkv.unlink(missing_ok=True)
                if len(dec) != len(frames):
                    print(f"  [!] {clip}: 디코딩 {len(dec)}/{len(frames)} — 잘린 만큼만 평가")
                    labs = labs[:len(dec)]
                frames = dec
            score(model, frames, labs, imgsz, a.conf, a.cls, acc)
            if (ci + 1) % 10 == 0:
                print(f"  [{name}] {ci+1}/{len(clips)} 클립", flush=True)
        rec = acc["tp"] / max(1, acc["tp"] + acc["fn"])
        fa = acc["fp_frames"] / max(1, acc["n_neg"])
        rows.append(dict(bitrate=name, kbps=kbps, n_pos=acc["n_pos"], n_neg=acc["n_neg"],
                         tp=acc["tp"], fn=acc["fn"], recall=round(rec, 4),
                         fp_frames=acc["fp_frames"], false_alarm=round(fa, 4),
                         mb=round(nbytes / 1e6, 1), sec=round(time.time() - t0, 1)))
        print(f"=== {name}: recall {rec:.4f} (TP {acc['tp']}/FN {acc['fn']}) | "
              f"오경보 {fa:.4f} ({acc['fp_frames']}/{acc['n_neg']}) | {rows[-1]['mb']}MB | "
              f"{rows[-1]['sec']}s", flush=True)

    p = Path(a.csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ {p}")
    base = rows[0]["recall"]
    print("원본 대비 recall 변화:")
    for r in rows[1:]:
        print(f"  {r['bitrate']:>10}: {r['recall']:.4f}  ({r['recall']-base:+.4f})  "
              f"오경보 {r['false_alarm']:.4f}")
    print("\n🔴 부분집합 결과다. 절대값을 전체 벤치마크(0.9755 등)와 직접 비교하지 말 것.")


if __name__ == "__main__":
    main()
