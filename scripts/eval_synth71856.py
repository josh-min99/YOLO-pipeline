"""
71856 합성 벤치마크 — 조건별(센서·계절·주야 / 해무 / 황천) 성능표.

synth71856_to_yolo.py 가 만든 <ds>/slices/*.yaml 을 하나씩 ultralytics val 로 돌린다.
목적은 "합성 도메인에서 기존 탐지기가 얼마나 무너지는가"를 재는 것이지, 합성으로
좋은 숫자를 만드는 게 아니다. 판정 기준은 어디까지나 W5 실데이터 홀드아웃이다.

두 지점에서 잰다:
  - mAP  : conf=0.001 (표준 mAP 정의)
  - 운용점: conf=0.6   (W5에서 정한 운용 conf) 의 P/R

사용:
    python scripts/eval_synth71856.py --weights best_spot.pt --ds datasets/synth71856
    python scripts/eval_synth71856.py --weights best_spot.pt --ds ... --only all,cond_
"""
import argparse, csv, json
from pathlib import Path

WARSHIP = 2   # marine.yaml: 0 fishing_boat / 1 merchant_ship / 2 warship


def run_slice(YOLO, weights, yaml_path, imgsz, device, conf, half):
    """ultralytics val 1회 -> dict. 클래스가 비어 있는 슬라이스도 죽지 않게 방어."""
    m = YOLO(weights).val(data=str(yaml_path), imgsz=imgsz, device=device,
                          conf=conf, half=half, verbose=False, plots=False,
                          save_json=False)
    b = m.box
    out = dict(mAP50=float(b.map50), mAP50_95=float(b.map),
               P=float(b.mp), R=float(b.mr))
    # 클래스별 — 존재하는 클래스만 채워진다(ap_class_index 로 매핑)
    for i, c in enumerate(b.ap_class_index):
        out[f"mAP50_cls{int(c)}"] = float(b.ap50[i])
        out[f"R_cls{int(c)}"] = float(b.r[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ds", required=True, help="synth71856_to_yolo.py 의 --out")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--op-conf", type=float, default=0.6, help="운용점 conf(W5 기준)")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--only", help="쉼표구분 접두사 필터(예: all,cond_)")
    ap.add_argument("--out", default="results/synth71856_conditions.csv")
    args = ap.parse_args()

    from ultralytics import YOLO   # 무거우니 인자 파싱 후에

    slice_dir = Path(args.ds) / "slices"
    yamls = sorted(slice_dir.glob("*.yaml"))
    if args.only:
        pref = tuple(s.strip() for s in args.only.split(","))
        yamls = [y for y in yamls if y.stem.startswith(pref)]
    assert yamls, f"슬라이스 없음: {slice_dir}"

    rows = []
    for y in yamls:
        n_img = len((y.with_suffix(".txt")).read_text(encoding="utf-8").splitlines())
        print(f"--- {y.stem} ({n_img} images)")
        r = dict(slice=y.stem, n_images=n_img)
        r.update({f"map_{k}": v for k, v in
                  run_slice(YOLO, args.weights, y, args.imgsz, args.device, 0.001, args.half).items()})
        r.update({f"op_{k}": v for k, v in
                  run_slice(YOLO, args.weights, y, args.imgsz, args.device, args.op_conf, args.half).items()})
        rows.append(r)
        print(f"    mAP50={r['map_mAP50']:.3f}  군함mAP50={r.get(f'map_mAP50_cls{WARSHIP}', float('nan')):.3f}  "
              f"운용점 P={r['op_P']:.3f} R={r['op_R']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r}, key=lambda c: (c != "slice", c != "n_images", c))
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\n저장: {out}")

    print("\n=== 요약 (군함 mAP50) ===")
    for r in rows:
        v = r.get(f"map_mAP50_cls{WARSHIP}")
        print(f"  {r['slice']:<28} n={r['n_images']:>5}  "
              f"{'-' if v is None else format(v, '.3f')}")


if __name__ == "__main__":
    main()
