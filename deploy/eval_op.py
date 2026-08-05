"""운영점 지표(recall@conf, 오경보율)를 predict로 직접 계산 — val()이 못 쓰는 rect 엔진용.

ultralytics val()은 imgsz=[h,w]를 정사각으로 되돌려 rect 엔진과 shape이 어긋난다(유의사항 §15-6).
검증: 이 스크립트의 정사각 recall 0.9725가 val의 R@0.6 0.9724와 일치(P1, 2026-07-29).

  python deploy/eval_op.py --weights engines/best_spot_736x1280_fp16.engine --imgsz 736x1280 \
      --data-root datasets/marine_session_spot --conf 0.6 --csv results/deploy/op_jetson.csv
"""
import argparse, csv, time
from pathlib import Path

def parse_imgsz(s):
    s = str(s).replace("x", ",")
    if "," in s:
        h, w = [int(v) for v in s.split(",")]; return [h, w]
    return int(s)

def load_labels(lab, W, H, cls):
    out = []
    for ln in Path(lab).read_text().splitlines():
        p = ln.split()
        if len(p) == 5 and int(p[0]) == cls:
            xc, yc, w, h = [float(v) for v in p[1:]]
            out.append([(xc-w/2)*W, (yc-h/2)*H, (xc+w/2)*W, (yc+h/2)*H])
    return out

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2-x1) * max(0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/ua if ua > 0 else 0

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--data-root", default="datasets/marine_session_spot")
ap.add_argument("--split", default="val")
ap.add_argument("--imgsz", default="1280")
ap.add_argument("--conf", type=float, default=0.6)
ap.add_argument("--cls", type=int, default=2)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--csv", default="", help="결과 1줄 append (없으면 헤더부터 생성)")
a = ap.parse_args()

root = Path(a.data_root)
labs = sorted((root/"labels"/a.split).glob("*.txt"))
if a.limit: labs = labs[:a.limit]
imgsz = parse_imgsz(a.imgsz)
from ultralytics import YOLO
m = YOLO(a.weights)
tp = fn = fp_frames = fp_boxes = n_pos = n_neg = 0
for i, lab in enumerate(labs):
    img = root/"images"/a.split/(lab.stem + ".jpg")
    if not img.exists(): continue
    r = m.predict(str(img), imgsz=imgsz, conf=a.conf, classes=[a.cls], verbose=False)[0]
    H, W = r.orig_shape
    gt = load_labels(lab, W, H, a.cls)
    pred = [[float(v) for v in b.xyxy[0].tolist()] for b in r.boxes]
    if gt:
        n_pos += 1
        used = set()
        for g in gt:
            hit = -1
            for j, p in enumerate(pred):
                if j not in used and iou(g, p) >= 0.5: hit = j; break
            if hit >= 0: used.add(hit); tp += 1
            else: fn += 1
    else:
        n_neg += 1
        if pred: fp_frames += 1; fp_boxes += len(pred)
    if (i+1) % 2000 == 0: print(f"  {i+1}/{len(labs)}", flush=True)

rec = tp/max(1, tp+fn)
fa = fp_frames/max(1, n_neg)
print(f"\n=== {Path(a.weights).name} @ {a.imgsz} conf={a.conf} ===")
print(f"군함 프레임 {n_pos} / 군함없는 프레임 {n_neg}")
print(f"recall@{a.conf} = {rec:.4f}  (TP {tp} / FN {fn})")
print(f"오경보율 = {fa:.4f}  ({fp_frames}프레임 / {fp_boxes}박스)")

if a.csv:
    p = Path(a.csv); p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "weights", "imgsz", "conf", "cls", "n_pos", "n_neg",
                        "tp", "fn", "recall", "fp_frames", "fp_boxes", "false_alarm"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), Path(a.weights).name, a.imgsz,
                    a.conf, a.cls, n_pos, n_neg, tp, fn, round(rec, 4),
                    fp_frames, fp_boxes, round(fa, 4)])
    print(f"→ {p}")
