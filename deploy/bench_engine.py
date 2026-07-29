"""
지연·FPS 측정 — 설정(정밀도 × 해상도)별 표를 만든다. P2 기종 선정의 근거 숫자.

  python deploy/bench_engine.py --weights best.pt,engines/best_1280_fp16.engine \
      --imgsz-list 1280,736x1280,960,640 --frames datasets/marine_session/images/val --n 200

측정 규율(W1 §8-3과 동일):
  - batch=1 (배포 조건), warmup 후 측정, GPU 비동기 때문에 **torch.cuda.synchronize()** 필수
  - 결과는 항상 CSV로 저장(§8-7 — 터미널 스크롤백은 날아간다)
  - 모델 내부 시간(pre/infer/post)과 벽시계 지연을 따로 본다: 전처리가 병목인 경우가 많다

출력의 핵심은 **'1280 정사각 대비 상대 연산비'** 열이다. 이 비율은 같은 네트워크라
보드가 바뀌어도 대체로 유지되므로, x86에서 재도 Jetson 기종 선정에 그대로 쓸 수 있다.
(절대 FPS는 보드에서 다시 재야 한다 — P3)
"""
import argparse
import csv
import platform
import time
from pathlib import Path

import numpy as np


def parse_imgsz(s):
    s = str(s).replace("x", ",")
    if "," in s:
        h, w = [int(v) for v in s.split(",")]
        return [h, w]
    return int(s)


def pixels(imgsz):
    return imgsz[0] * imgsz[1] if isinstance(imgsz, list) else imgsz * imgsz


def load_frames(spec, k):
    """벤치용 실제 프레임 k장(합성 노이즈로 재면 전처리·NMS 부하가 비현실적으로 낮게 나온다)."""
    import cv2
    p = Path(spec)
    if p.is_dir():
        files = sorted(p.glob("*.jpg"))[:k]
        if not files:
            raise SystemExit(f"이미지 없음: {p}")
        return [cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR) for f in files]
    cap = cv2.VideoCapture(str(p))
    frames = []
    while len(frames) < k:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"프레임을 못 읽음: {p}")
    return frames


def bench(weights, imgsz, frames, conf, device, warmup, n, half):
    import torch
    from ultralytics import YOLO

    model = YOLO(weights)
    # half는 필요할 때만 넘긴다 — ultralytics 8.4에서 deprecated 경고가 프레임마다 찍힘
    kw = dict(imgsz=imgsz, conf=conf, device=device, verbose=False)
    if half:
        kw["half"] = True
    for i in range(warmup):
        model.predict(frames[i % len(frames)], **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    lat, sp = [], []
    for i in range(n):
        f = frames[i % len(frames)]
        t0 = time.perf_counter()
        r = model.predict(f, **kw)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e3)
        sp.append(r.speed)
    a = np.array(lat)
    return dict(mean=a.mean(), p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                fps=1000.0 / a.mean(),
                pre=float(np.mean([s["preprocess"] for s in sp])),
                inf=float(np.mean([s["inference"] for s in sp])),
                post=float(np.mean([s["postprocess"] for s in sp])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="콤마 구분: best.pt,engines/xx_fp16.engine,...")
    ap.add_argument("--imgsz-list", default="1280,736x1280,960,640")
    ap.add_argument("--frames", default="datasets/marine_session/images/val",
                    help="실제 프레임 폴더 또는 비디오")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true", help=".pt를 FP16으로 (엔진은 빌드 시 정밀도 고정)")
    ap.add_argument("--out", default="results/deploy")
    args = ap.parse_args()

    frames = load_frames(args.frames, max(args.warmup, 32))
    print(f"벤치 프레임 {len(frames)}장 ({frames[0].shape[1]}x{frames[0].shape[0]})")

    rows = []
    for w in [x.strip() for x in args.weights.split(",") if x.strip()]:
        if not Path(w).exists():
            print(f"[skip] 없음: {w}")
            continue
        for s in [x.strip() for x in args.imgsz_list.split(",") if x.strip()]:
            imgsz = parse_imgsz(s)
            print(f"\n--- {Path(w).name} @ {s} ---")
            try:
                r = bench(w, imgsz, frames, args.conf, args.device, args.warmup, args.n, args.half)
            except Exception as e:
                print(f"[fail] {type(e).__name__}: {e}")
                continue
            r.update(weights=Path(w).name, imgsz=s, px=pixels(imgsz))
            rows.append(r)
            print(f"p50 {r['p50']:.2f}ms  p95 {r['p95']:.2f}ms  {r['fps']:.1f} FPS "
                  f"(pre {r['pre']:.2f} / infer {r['inf']:.2f} / post {r['post']:.2f})")

    if not rows:
        raise SystemExit("측정된 조합이 없음")

    # 1280 정사각 기준 상대 연산비 (없으면 가장 느린 조합 기준)
    # ⚠️ 빠른 dGPU(3090 등)에서는 전처리·NMS(CPU)가 섞인 wall-clock 비율이 압축돼 보인다.
    #    보드 예측에 쓸 값은 **모델 내부 inference 시간 기준(rel_cost_infer)** 이다.
    base = next((r for r in rows if r["imgsz"] == "1280"), max(rows, key=lambda r: r["mean"]))
    for r in rows:
        r["rel_cost"] = r["mean"] / base["mean"]
        r["rel_cost_infer"] = r["inf"] / base["inf"]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"bench_{platform.node()}_{time.strftime('%m%d_%H%M')}.csv"
    cols = ["weights", "imgsz", "px", "p50", "p95", "mean", "fps", "pre", "inf", "post",
            "rel_cost", "rel_cost_infer"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in cols})

    print(f"\n{'가중치':<34}{'imgsz':>10}{'p50(ms)':>10}{'infer':>8}{'FPS':>8}{'상대비':>8}{'(infer)':>9}")
    for r in sorted(rows, key=lambda r: r["mean"]):
        print(f"{r['weights']:<34}{r['imgsz']:>10}{r['p50']:>10.2f}{r['inf']:>8.2f}"
              f"{r['fps']:>8.1f}{r['rel_cost']:>8.2f}{r['rel_cost_infer']:>9.2f}")
    print(f"\nCSV: {csv_path}")
    print("※ W6 계획서 §P2 결정표에는 **rel_cost_infer**(모델 내부 기준)를 넣는다.")
    print("   빠른 dGPU에선 wall-clock(rel_cost)이 CPU 전처리·NMS에 눌려 차이가 작게 보인다.")
    print("※ 절대 FPS는 보드에서 다시 측정(P3). end-to-end 는 deploy/stream_infer.py 리포트로.")


if __name__ == "__main__":
    main()
