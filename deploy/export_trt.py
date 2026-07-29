"""
best.pt → ONNX / TensorRT 엔진(FP16·INT8) 변환.

  # x86(vast)에서 설정 탐색: 정사각/rect × FP16/INT8
  python deploy/export_trt.py --best best.pt --imgsz 1280      --formats onnx,fp16,int8
  python deploy/export_trt.py --best best.pt --imgsz 736,1280  --formats onnx,fp16,int8

  # Jetson에서는 ONNX만 옮겨와서 엔진을 다시 빌드
  python deploy/export_trt.py --best best.onnx --imgsz 736,1280 --formats fp16,int8

🔴 함정 1 — 엔진은 이식 불가.
   .engine 은 GPU 아키텍처·TensorRT 버전에 종속이라 x86에서 만든 걸 Jetson에 복사하면 안 돈다.
   **ONNX만 옮기고 엔진은 타깃 보드에서 다시 빌드**한다. (x86 결과는 '설정 탐색'용)

🔴 함정 2 — INT8 캘리브레이션 데이터.
   ultralytics 는 `data:` yaml의 **val 스플릿**으로 캘리브레이션한다. 그대로 쓰면 평가셋을
   양자화 튜닝에 쓰는 셈이라 W5에서 세운 무누수 규율이 깨진다.
   → 이 스크립트는 val 자리에 **train 이미지**를 넣은 임시 yaml을 만들어서 넘긴다(--calib-split).

🔴 함정 3 — 변환 후 정확도 재측정 생략 금지.
   INT8은 클래스마다 손실이 다르고, 특히 수평선의 소형 표적에서 recall이 먼저 떨어진다.
   변환이 끝나면 반드시 deploy/eval_engine.py 로 패리티 표를 채울 것.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml


def parse_imgsz(s):
    if "," in str(s):
        h, w = [int(v) for v in str(s).split(",")]
        return [h, w]
    return int(s)


def make_calib_yaml(data_yaml, split, out_path):
    """캘리브레이션용 yaml: val 자리에 train(또는 지정 스플릿) 이미지를 넣는다."""
    src = Path(data_yaml)
    d = yaml.safe_load(src.read_text(encoding="utf-8"))
    # path가 상대경로면 원본 yaml 기준으로 절대화 — 임시 yaml을 다른 폴더에 쓰면 깨지므로
    d["path"] = str((src.parent / d["path"]).resolve()) if not Path(d["path"]).is_absolute() \
        else d["path"]
    d["val"] = f"images/{split}"
    Path(out_path).write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    print(f"[calib] {out_path}  (val ← images/{split}, 평가셋 누수 방지)")
    return str(out_path)


def sha8(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True, help="best.pt (또는 보드에서 재빌드 시 .onnx)")
    ap.add_argument("--imgsz", default="1280", help="1280 또는 736,1280(rect)")
    ap.add_argument("--formats", default="onnx,fp16", help="onnx,fp16,int8 중 콤마 조합")
    ap.add_argument("--data", default="configs/marine.yaml", help="INT8 캘리브레이션용 data yaml")
    ap.add_argument("--calib-split", default="train", help="캘리브레이션 이미지 스플릿(기본 train)")
    ap.add_argument("--calib-fraction", type=float, default=0.02, help="캘리브레이션 사용 비율(≈수백 장)")
    ap.add_argument("--batch", type=int, default=1, help="배포는 batch=1")
    ap.add_argument("--workspace", type=float, default=4.0, help="TRT workspace GB")
    ap.add_argument("--device", default="0")
    ap.add_argument("--outdir", default="engines")
    ap.add_argument("--nms", action="store_true", help="NMS를 엔진에 포함(지연 측정 기준이 달라짐 — 표에 명시)")
    args = ap.parse_args()

    if not Path(args.best).exists():
        raise SystemExit(
            f"가중치 없음: {args.best}\n"
            "  가중치는 git에 없다(용량). Jupyter 파일 브라우저로 marine_spot_results.tgz 업로드 후\n"
            "  tar -xzf marine_spot_results.tgz  → marine_spot_yolo11s_1280/weights/best.pt")

    imgsz = parse_imgsz(args.imgsz)
    tag_sz = f"{imgsz[0]}x{imgsz[1]}" if isinstance(imgsz, list) else f"{imgsz}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    from ultralytics import YOLO
    import torch
    import ultralytics

    meta = dict(source=str(args.best), source_sha8=sha8(args.best), imgsz=imgsz,
                batch=args.batch, nms=args.nms,
                ultralytics=ultralytics.__version__, torch=torch.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                built=time.strftime("%Y-%m-%d %H:%M:%S"), outputs={})
    try:
        import tensorrt
        meta["tensorrt"] = tensorrt.__version__
    except Exception:
        meta["tensorrt"] = None

    common = dict(imgsz=imgsz, batch=args.batch, device=args.device, nms=args.nms, verbose=False)

    for fmt in formats:
        model = YOLO(args.best)          # export 는 모델 상태를 바꾸므로 매번 새로 로드
        t0 = time.time()
        if fmt == "onnx":
            print(f"\n=== ONNX {tag_sz} ===")
            out = model.export(format="onnx", simplify=True, dynamic=False, opset=17, **common)
        elif fmt in ("fp16", "engine", "engine-fp16"):
            print(f"\n=== TensorRT FP16 {tag_sz} ===")
            out = model.export(format="engine", half=True, workspace=args.workspace,
                               dynamic=False, **common)
        elif fmt in ("int8", "engine-int8"):
            print(f"\n=== TensorRT INT8 {tag_sz} (calib={args.calib_split}) ===")
            calib = make_calib_yaml(args.data, args.calib_split, outdir / "calib_data.yaml")
            out = model.export(format="engine", int8=True, data=calib,
                               fraction=args.calib_fraction, workspace=args.workspace,
                               dynamic=False, **common)
        elif fmt == "fp32":
            print(f"\n=== TensorRT FP32 {tag_sz} ===")
            out = model.export(format="engine", half=False, workspace=args.workspace,
                               dynamic=False, **common)
        else:
            print(f"[skip] 모르는 포맷: {fmt}")
            continue

        src = Path(out)
        dst = outdir / f"{Path(args.best).stem}_{tag_sz}_{fmt}{src.suffix}"
        shutil.move(str(src), dst)
        size_mb = dst.stat().st_size / 1e6
        meta["outputs"][fmt] = dict(path=str(dst), size_mb=round(size_mb, 1),
                                    build_s=round(time.time() - t0, 1))
        print(f"→ {dst}  ({size_mb:.1f} MB, {time.time()-t0:.0f}s)")

    (outdir / f"meta_{Path(args.best).stem}_{tag_sz}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n다음:")
    print(f"  1) 정확도 패리티  python deploy/eval_engine.py --weights {args.best},"
          f"{outdir}/*_{tag_sz}_fp16.engine --imgsz {args.imgsz}")
    print(f"  2) 지연·FPS      python deploy/bench_engine.py --weights <위와 동일> --imgsz {args.imgsz}")
    print("  3) 보드로는 .onnx 만 복사해서 거기서 다시 이 스크립트로 엔진 빌드(엔진은 이식 불가)")


if __name__ == "__main__":
    main()
