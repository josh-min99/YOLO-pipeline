"""
정확도 패리티 표 — 엔진을 바꿔도 성능이 유지되는가. **변환 후 이걸 안 재면 배포 근거가 없다.**

  python deploy/eval_engine.py \
      --weights runs/.../best.pt,engines/best_1280_fp16.engine,engines/best_1280_int8.engine \
      --data configs/marine.yaml --data-root datasets/marine_session_spot --imgsz 1280

내는 표(W5 기준값과 같은 축):
  엔진 | 군함 mAP50 | 전체 mAP50 | 운영점(0.6) 군함 P/R | 빈프레임 오경보율 | (지연은 bench_engine.py)

기준(W5, PyTorch FP32, spot 홀드아웃): 군함 mAP50 0.979 · 전체 0.961 · 운영점 P0.951 R0.975 · 오경보 0.2%
합격선(제안): 군함 mAP50 손실 FP16 ≤1%p / INT8 ≤2%p, **군함 recall@0.6 ≥ 0.95**, 오경보 ≤0.5%

⚠️ conf=0.6 으로 val 한 mAP는 잘린 값이라 표준 mAP와 섞어 보고하면 안 된다(W5에서 한 번 헷갈렸던 지점).
   표준 mAP는 conf≈0.001, 운영점 P/R은 conf=0.6 — 열을 분리해서 낸다.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_warship_fp import collect_negatives, count_false_alarms  # noqa: E402


def parse_imgsz(s):
    s = str(s).replace("x", ",")
    if "," in s:
        h, w = [int(v) for v in s.split(",")]
        return [h, w]
    return int(s)


def per_class(metrics, names, warship=2):
    """클래스별 mAP50 dict. ultralytics 는 등장한 클래스만 ap50 배열로 준다."""
    out = {}
    try:
        for i, ci in enumerate(metrics.box.ap_class_index):
            out[names.get(int(ci), int(ci))] = float(metrics.box.ap50[i])
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="콤마 구분. 첫 번째가 기준(보통 best.pt)")
    ap.add_argument("--data", default="configs/marine.yaml")
    ap.add_argument("--data-root", default="", help="오경보 측정용 데이터셋 루트(생략 시 data yaml의 path)")
    ap.add_argument("--imgsz", default="1280")
    ap.add_argument("--split", default="val")
    ap.add_argument("--op-conf", type=float, default=0.6, help="운영점")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warship-class", type=int, default=2)
    ap.add_argument("--out", default="results/deploy")
    args = ap.parse_args()

    import yaml
    from ultralytics import YOLO

    imgsz = parse_imgsz(args.imgsz)
    dcfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = {int(k): v for k, v in dcfg["names"].items()}
    wname = names.get(args.warship_class, "warship")
    root = args.data_root or dcfg["path"]

    negs = collect_negatives(root, args.split, args.warship_class)
    print(f"[오경보 측정] 군함 없는 {args.split} 프레임 {len(negs)}장 @ {root}")

    rows = []
    for w in [x.strip() for x in args.weights.split(",") if x.strip()]:
        if not Path(w).exists():
            print(f"[skip] 없음: {w}")
            continue
        print(f"\n=== {Path(w).name} ===")
        r = dict(weights=Path(w).name, imgsz=args.imgsz)
        try:
            model = YOLO(w)
            # (1) 표준 mAP — conf 매우 낮게
            m = model.val(data=args.data, imgsz=imgsz, split=args.split, conf=0.001,
                          batch=args.batch, verbose=False)
            pc = per_class(m, names)
            r["map50"] = round(float(m.box.map50), 4)
            r["map50_95"] = round(float(m.box.map), 4)
            r["warship_map50"] = round(pc.get(wname, float("nan")), 4)
            for k, v in pc.items():
                r[f"ap50_{k}"] = round(v, 4)

            # (2) 운영점 P/R — conf=0.6 (이 값의 mAP는 잘린 값이므로 쓰지 않는다)
            m2 = model.val(data=args.data, imgsz=imgsz, split=args.split, conf=args.op_conf,
                           batch=args.batch, verbose=False)
            idx = list(m2.box.ap_class_index).index(args.warship_class) \
                if args.warship_class in list(m2.box.ap_class_index) else None
            if idx is not None:
                r["op_P_warship"] = round(float(m2.box.p[idx]), 4)
                r["op_R_warship"] = round(float(m2.box.r[idx]), 4)

            # (3) 빈 프레임 오경보율
            # 🔴 TRT 엔진은 batch=1 고정으로 빌드되므로 predict에 리스트를 묶어 넣으면
            #    AssertionError(input size [8,3,H,W] != max model size [1,3,H,W]). 엔진이면 1로.
            pb = 1 if str(w).endswith(".engine") else args.batch
            t0 = time.time()
            fp_frames, fp_boxes = count_false_alarms(model, negs, args.op_conf,
                                                     args.warship_class, imgsz, pb)
            r["fp_rate"] = round(fp_frames / max(1, len(negs)), 4)
            r["fp_frames"] = fp_frames
            r["fp_boxes"] = fp_boxes
            print(f"군함 mAP50 {r['warship_map50']} / 전체 {r['map50']} / "
                  f"운영점 P{r.get('op_P_warship')} R{r.get('op_R_warship')} / "
                  f"오경보 {r['fp_rate']:.2%} ({time.time()-t0:.0f}s)")
        except Exception as e:
            # 부분 결과는 살린다 — mAP까지 재놓고 오탐 단계에서 죽으면 다시 30분 걸린다
            print(f"[fail] {type(e).__name__}: {e}")
            r["error"] = f"{type(e).__name__}: {e}"
            if "warship_map50" in r:
                print(f"  (부분 결과 보존) 군함 mAP50 {r['warship_map50']} / 전체 {r.get('map50')}")
        rows.append(r)

    if not rows:
        raise SystemExit("평가된 가중치가 없음")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r})
    csv_path = outdir / f"parity_{time.strftime('%m%d_%H%M')}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    base = rows[0]
    print(f"\n{'엔진':<34}{'군함mAP50':>10}{'Δ':>8}{'전체':>8}{'R@0.6':>8}{'오경보':>9}")
    for r in rows:
        if "error" in r and "warship_map50" not in r:
            print(f"{r['weights']:<34}  {r['error']}")
            continue
        d = r.get("warship_map50", 0) - base.get("warship_map50", 0)
        print(f"{r['weights']:<34}{r.get('warship_map50', 0):>10.4f}{d:>+8.4f}"
              f"{r.get('map50', 0):>8.4f}{r.get('op_R_warship', 0):>8.3f}{r.get('fp_rate', 0):>8.2%}")
    print(f"\nCSV: {csv_path}")
    print("판정: 군함 mAP50 손실 FP16 ≤0.01 / INT8 ≤0.02, R@0.6 ≥0.95, 오경보 ≤0.5% 면 통과.")
    print("      통과한 설정 중 bench_engine.py 의 상대비가 가장 낮은 것 = 배포 설정.")


if __name__ == "__main__":
    main()
