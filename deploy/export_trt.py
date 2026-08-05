"""
best.pt → ONNX / TensorRT 엔진(FP16·INT8) 변환.

  # x86(vast)에서 설정 탐색: 정사각/rect × FP16/INT8
  python deploy/export_trt.py --best best.pt --imgsz 1280      --formats onnx,fp16,int8
  python deploy/export_trt.py --best best.pt --imgsz 736,1280  --formats onnx,fp16,int8

  # Jetson에서는 .pt 를 옮겨와 보드에서 엔진을 다시 빌드 (권장 경로)
  python3 deploy/export_trt.py --best best_spot.pt --imgsz 736,1280 --formats fp16

  # ONNX만 있을 때(폴백) — ultralytics를 안 거치고 trtexec으로 직접 빌드
  python3 deploy/export_trt.py --best best_spot_736x1280_onnx.onnx --imgsz 736,1280 --formats fp16

🔴 함정 1 — 엔진은 이식 불가.
   .engine 은 GPU 아키텍처·TensorRT 버전에 종속이라 x86에서 만든 걸 Jetson에 복사하면 안 돈다.
   **엔진은 타깃 보드에서 다시 빌드**한다. (x86 결과는 '설정 탐색'용)

🔴 함정 1-b — 그런데 **ONNX를 ultralytics로 재빌드할 수는 없다.**
   `Model.export()` 는 첫 줄에서 `_check_is_pytorch_model()` 을 호출해서
   `TypeError: model='...onnx' should be a *.pt PyTorch model` 로 죽는다.
   → **.pt 를 보드로 옮기는 것이 정답**이다(19MB, ONNX보다 작다). 보드에서 .pt로 export하면
     클래스명·stride 메타데이터가 엔진에 박혀서 stream_infer 의 라벨이 정상으로 나온다.
   → ONNX밖에 없으면 이 스크립트가 **trtexec 으로 우회**한다. 단 그렇게 만든 엔진에는
     ultralytics 메타데이터가 없어서 클래스명이 `class0/1/2` 로 폴백된다(인덱스는 정상).

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
import subprocess
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


def find_trtexec():
    """JetPack은 /usr/src/tensorrt/bin/trtexec 에 둔다(PATH에 없음)."""
    for c in ("/usr/src/tensorrt/bin/trtexec", "/opt/tensorrt/bin/trtexec"):
        if Path(c).is_file():
            return c
    return shutil.which("trtexec")


def build_with_trtexec(onnx, imgsz, out, half, workspace_gb, input_name="images"):
    """ONNX → .engine (ultralytics 미경유). 메타데이터가 없는 엔진이 나온다 — 함정 1-b 참고."""
    exe = find_trtexec()
    if not exe:
        raise SystemExit(
            "trtexec 없음. JetPack(TensorRT) 설치를 확인하거나 .pt 를 옮겨와 --best best.pt 로 빌드할 것.\n"
            "  점검: bash deploy/jetson/setup.sh")
    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    cmd = [exe, f"--onnx={onnx}", f"--saveEngine={out}",
           f"--shapes={input_name}:1x3x{h}x{w}", f"--memPoolSize=workspace:{int(workspace_gb*1024)}"]
    if half:
        cmd.append("--fp16")
    print("[trtexec] " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0 or not Path(out).exists():
        raise SystemExit(
            f"trtexec 실패(코드 {r.returncode}).\n"
            "  자주 걸리는 것: ① 입력 텐서 이름이 'images'가 아님 → --input-name 으로 지정\n"
            "                 ② workspace 부족 → --workspace 를 낮출 것(Orin Nano 8GB는 2 권장)")
    return out


def onnx_route(args, imgsz, tag_sz, outdir, formats):
    """--best 가 .onnx 일 때: ultralytics를 못 거치므로 trtexec 으로 직접 빌드."""
    print("[!] ONNX 입력 - ultralytics export는 .pt만 받는다(함정 1-b). trtexec으로 빌드한다.")
    print("    나온 엔진에는 클래스명 메타데이터가 없어 stream_infer 라벨이 class0/1/2 로 뜬다.")
    print("    라벨까지 정상으로 원하면 best_spot.pt 를 옮겨와 --best best_spot.pt 로 다시 빌드할 것.\n")

    meta = dict(source=str(args.best), source_sha8=sha8(args.best), imgsz=imgsz,
                batch=args.batch, nms=args.nms, builder="trtexec", metadata_in_engine=False,
                built=time.strftime("%Y-%m-%d %H:%M:%S"), outputs={})
    for fmt in formats:
        if fmt == "onnx":
            continue                      # 이미 ONNX다
        if fmt in ("int8", "engine-int8"):
            print("[skip] INT8은 캘리브레이션 데이터가 필요해 trtexec 갈래에서 지원 안 함 "
                  "— .pt 로 빌드할 것(--formats int8)")
            continue
        if fmt not in ("fp16", "fp32", "engine", "engine-fp16"):
            print(f"[skip] 모르는 포맷: {fmt}")
            continue
        half = fmt != "fp32"
        dst = outdir / f"{Path(args.best).stem}_{tag_sz}_{fmt}.engine"
        t0 = time.time()
        build_with_trtexec(args.best, imgsz, str(dst), half, args.workspace, args.input_name)
        meta["outputs"][fmt] = dict(path=str(dst), size_mb=round(dst.stat().st_size / 1e6, 1),
                                    build_s=round(time.time() - t0, 1))
        print(f"→ {dst}  ({dst.stat().st_size/1e6:.1f} MB, {time.time()-t0:.0f}s)")

    (outdir / f"meta_{Path(args.best).stem}_{tag_sz}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n다음: deploy/bench_engine.py 로 지연·FPS, deploy/eval_op.py 로 운영점 지표.")


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
    ap.add_argument("--input-name", default="images", help="ONNX 입력 텐서 이름(trtexec 폴백용)")
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

    # ONNX 입력이면 ultralytics를 못 쓴다(함정 1-b) → trtexec 갈래로 빠진다.
    if Path(args.best).suffix.lower() == ".onnx":
        return onnx_route(args, imgsz, tag_sz, outdir, formats)

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
    print("  3) 보드로는 **.pt** 를 복사해서 거기서 다시 이 스크립트로 엔진 빌드(엔진은 이식 불가,\n"
          "     ONNX는 ultralytics로 재빌드가 안 돼서 trtexec 폴백 = 메타데이터 없는 엔진이 된다)")


if __name__ == "__main__":
    main()
