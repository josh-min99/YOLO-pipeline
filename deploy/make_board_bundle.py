"""보드(Jetson)로 USB 이전할 번들을 만든다 — 가중치 + **동일 벤치마크** + 데모 + 실행 지침.

  # 데이터셋이 있는 호스트(vast 또는 로컬)에서
  python deploy/make_board_bundle.py --root datasets/marine_session_spot \
      --weights best_spot.pt --demo demo_I2_S0_C5_0079_in.avi --out /media/usb/jetson_bundle

  # 빠른 리허설(이미지 200장만) — 배관 확인용, 벤치마크 숫자로는 쓰지 말 것
  python deploy/make_board_bundle.py --root ... --out ... --limit 200

왜 이 스크립트인가:
  이번 단계의 비교는 **x86(P1) vs 보드**다. 같은 val이어야 "TRT FP16 손실 0"이나 "recall 0.9755"가
  보드 숫자와 같은 축에 놓인다. 보드에서 다른 val을 쓰면 그 기준선이 끊긴다.
  그래서 이 스크립트는 복사만 하는 게 아니라 **DATASET.md §6-1의 지문(프레임·박스·클래스 수)을
  대조**해서 다르면 경고한다. (기준 val = `splits_session_spot` 13,020장, 새 장소 홀드아웃)

🔴 엔진(.engine)은 넣지 않는다 — 아키텍처·TRT 버전 종속이라 보드에서 재빌드해야 한다(§15-1).
🔴 .pt 를 넣는다 — ONNX는 ultralytics로 재빌드가 안 된다(export_trt.py 함정 1-b).
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

# DATASET.md §6-1 — 이 숫자가 "같은 벤치마크"의 지문이다.
EXPECT = dict(frames=13020, boxes=14272, per_class={0: 7919, 1: 1910, 2: 4443})
NAMES = {0: "fishing_boat", 1: "merchant_ship", 2: "warship"}


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def make_resolver(root, split, images_dir):
    """라벨 stem → 이미지 경로. images_dir 를 주면 평면 폴더(datasets/marine_frames)에서 찾는다.

    이미지가 없는 라벨은 None — 지문도 복사도 **둘 다 이걸 기준으로** 센다.
    (json_to_yolo --labels-only 는 이미지 없는 프레임의 라벨도 쓰기 때문에
     라벨만 세면 val 이 13,022 로 나와 기준값 13,020 과 어긋난다.)
    """
    base = Path(images_dir) if images_dir else root / "images" / split

    def resolve(stem):
        p = base / f"{stem}.jpg"
        return p if p.exists() else None
    return resolve


def scan_labels(lab_dir, resolve):
    """프레임 수·박스 수·클래스별 인스턴스 — 벤치마크 동일성 지문(이미지 있는 것만)."""
    frames = boxes = missing = 0
    per_class = {}
    for t in sorted(lab_dir.glob("*.txt")):
        if resolve(t.stem) is None:
            missing += 1
            continue
        frames += 1
        for ln in t.read_text().splitlines():
            p = ln.split()
            if len(p) == 5:
                boxes += 1
                c = int(p[0])
                per_class[c] = per_class.get(c, 0) + 1
    return frames, boxes, per_class, missing


def copy_split(root, out, split, resolve, limit=0):
    """images/<split>, labels/<split> 로 복사. 심링크는 실체를 따라간다(--link 데이터셋 대비)."""
    dst_img, dst_lab = out / "images" / split, out / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lab.mkdir(parents=True, exist_ok=True)

    labs = sorted((root / "labels" / split).glob("*.txt"))
    if limit:
        labs = [t for t in labs if resolve(t.stem)][:limit]
    n_img = total = 0
    for i, lab in enumerate(labs):
        img = resolve(lab.stem)
        if img is None:
            continue
        dst = dst_img / f"{lab.stem}.jpg"
        shutil.copy2(img, dst)                    # copy2는 심링크를 따라가 실체를 복사
        shutil.copy2(lab, dst_lab / lab.name)
        total += dst.stat().st_size
        n_img += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(labs)}  ({total/1e9:.2f} GB)", flush=True)
    return n_img, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/marine_session_spot",
                    help="YOLO 데이터셋 루트(images/, labels/) — 반드시 spot 홀드아웃")
    ap.add_argument("--split", default="val")
    ap.add_argument("--images-dir", default="",
                    help="평면 이미지 폴더(datasets/marine_frames). 주면 root/images/ 대신 여기서 찾는다 "
                         "— json_to_yolo --labels-only 로 라벨만 만든 경우")
    ap.add_argument("--weights", default="best_spot.pt", help="보드로 옮길 .pt (엔진은 보드에서 빌드)")
    ap.add_argument("--onnx", default="", help="폴백용 ONNX(선택). trtexec 경로에서만 쓴다")
    ap.add_argument("--demo", default="", help="데모 클립(.avi/.mp4) — e2e FPS 측정용")
    ap.add_argument("--out", required=True, help="USB 마운트 경로 등 번들 생성 위치")
    ap.add_argument("--limit", type=int, default=0, help="이미지 N장만(리허설용, 벤치마크 무효)")
    ap.add_argument("--skip-images", action="store_true", help="가중치·문서만 갱신")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    if not (root / "labels" / args.split).is_dir():
        raise SystemExit(f"데이터셋 없음: {root}/labels/{args.split}\n"
                         "  DATASET.md §5 로 datasets/marine_session_spot 를 먼저 만들 것")
    out.mkdir(parents=True, exist_ok=True)
    bench = out / "benchmark" / "marine_session_spot"
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), source_root=str(root.resolve()),
               split=args.split, limit=args.limit, files={}, benchmark={})

    # 1) 벤치마크 지문 검사 — 복사 전에 원본부터 본다
    resolve = make_resolver(root, args.split, args.images_dir)
    print(f"=== 1. 벤치마크 지문 대조 ({root}/labels/{args.split}) ===")
    if args.images_dir:
        print(f"  이미지 출처: {args.images_dir} (평면 폴더)")
    frames, boxes, per_class, no_img = scan_labels(root / "labels" / args.split, resolve)
    print(f"  frames={frames}  boxes={boxes}  " +
          "  ".join(f"{NAMES.get(c,c)}={per_class.get(c,0)}" for c in sorted(per_class)))
    if no_img:
        print(f"  이미지 없는 라벨 {no_img}개 제외 (zip 손상분 2개면 정상)")
    same = (frames == EXPECT["frames"] and boxes == EXPECT["boxes"]
            and all(per_class.get(c, 0) == n for c, n in EXPECT["per_class"].items()))
    if same:
        print("  OK - x86(P1) 기준선과 동일한 벤치마크")
    else:
        exp = ", ".join(f"{NAMES[c]}={n}" for c, n in EXPECT["per_class"].items())
        print(f"  [!] 다름 - 기대값 frames={EXPECT['frames']} boxes={EXPECT['boxes']} ({exp})")
        print("  [!] 분할을 재생성했거나 이미지 손실이 다르다. 이 상태로 낸 숫자는 "
              "x86 기준선 0.9788/0.9755 와 직접 비교 불가(DATASET.md §6-1).")
    man["benchmark"] = dict(frames=frames, boxes=boxes,
                            per_class={NAMES.get(c, str(c)): n for c, n in sorted(per_class.items())},
                            matches_reference=same, reference=EXPECT["frames"])

    # 2) 이미지·라벨
    if args.skip_images:
        print("\n=== 2. 이미지 복사 건너뜀(--skip-images) ===")
    else:
        print(f"\n=== 2. {args.split} 복사 → {bench} ===")
        n_img, total = copy_split(root, bench, args.split, resolve, args.limit)
        print(f"  이미지·라벨 {n_img}쌍 / {total/1e9:.2f} GB")
        man["files"]["images"] = dict(count=n_img, bytes=total)

    # 3) 가중치 (+ 선택 ONNX)
    print("\n=== 3. 가중치 ===")
    wdir = out / "weights"
    wdir.mkdir(exist_ok=True)
    for tag, src in (("weights", args.weights), ("onnx", args.onnx)):
        if not src:
            continue
        s = Path(src)
        if not s.exists():
            print(f"  [!] 없음, 건너뜀: {s}")
            continue
        shutil.copy2(s, wdir / s.name)
        d = sha256(s)
        man["files"][tag] = dict(name=s.name, sha256=d, sha8=d[:8],
                                 bytes=s.stat().st_size)
        print(f"  {s.name}  sha8={d[:8]}  {s.stat().st_size/1e6:.1f} MB")

    if args.demo and Path(args.demo).exists():
        ddir = out / "demo"
        ddir.mkdir(exist_ok=True)
        shutil.copy2(args.demo, ddir / Path(args.demo).name)
        man["files"]["demo"] = dict(name=Path(args.demo).name,
                                    bytes=Path(args.demo).stat().st_size)
        print(f"  데모 {Path(args.demo).name}")

    # 4) 보드용 data yaml + 경로 고정 스크립트
    #    ultralytics 는 상대 path 를 yaml 위치가 아니라 DATASETS_DIR 기준으로 푼다 → 절대경로로 박아야 한다.
    (out / "marine_board.yaml").write_text(
        "# 보드용 data config — path 는 setup_bundle.sh 가 실제 위치로 덮어쓴다.\n"
        f"path: __BUNDLE__/benchmark/marine_session_spot\n"
        "train: images/val   # 보드에선 학습 안 함(자리만 채움)\n"
        f"val: images/{args.split}\n\nnames:\n"
        + "".join(f"  {c}: {n}\n" for c, n in NAMES.items()), encoding="utf-8")
    (out / "setup_bundle.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# 번들을 푼 위치로 data yaml 의 path 를 고정한다. 보드에서 제일 먼저 1회 실행.\n"
        "set -e\n"
        'HERE="$(cd "$(dirname "$0")" && pwd)"\n'
        'sed -i "s|^path:.*|path: $HERE/benchmark/marine_session_spot|" "$HERE/marine_board.yaml"\n'
        'grep "^path:" "$HERE/marine_board.yaml"\n'
        'echo "OK - 다음: RUN_ON_BOARD.md"\n', encoding="utf-8")
    (out / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    (out / "RUN_ON_BOARD.md").write_text(board_readme(args, man), encoding="utf-8")

    print(f"\n=== 완료 → {out.resolve()} ===")
    print("  MANIFEST.json / RUN_ON_BOARD.md / setup_bundle.sh 생성됨")
    if args.limit:
        print("  [!] --limit 을 썼다 = 벤치마크가 아니라 리허설 번들이다")


def board_readme(args, man):
    w = man["files"].get("weights", {}).get("name", "best_spot.pt")
    sha8 = man["files"].get("weights", {}).get("sha8", "?")
    b = man["benchmark"]
    demo = man["files"].get("demo", {}).get("name", "<clip>")
    ok = "동일 ✅" if b.get("matches_reference") else "**다름 ⚠️ — 기존 숫자와 직접 비교 불가**"
    return f"""# 보드에서 실행하기 (USB 번들)

> 생성 {man['created']} · 가중치 `{w}` sha8=`{sha8}` · 벤치마크 {b['frames']}장 / {b['boxes']}박스 → {ok}
> 기준선(RTX 3090, W6 P1): 군함 mAP50 **0.9788** · rect recall@0.6 **0.9755** · 오경보 **2.12%**
> · 추론 2.25ms · 헤드리스 e2e p50 **9.43ms(~106 FPS)**

## 0. 번들 고정 (1회)
```bash
bash setup_bundle.sh          # marine_board.yaml 의 path 를 실제 위치로 덮어씀
```

## 1. 환경
```bash
git clone https://github.com/josh-min99/YOLO-pipeline.git && cd YOLO-pipeline
bash deploy/jetson/setup.sh                  # JetPack·TRT·전력모드 점검
```
🔴 `pip install ultralytics` 를 그냥 하면 **CPU torch** 가 깔린다(§15-3). 컨테이너를 쓸 것:
```bash
sudo docker run -it --rm --runtime nvidia --network host \\
  -v $(pwd):/work -v <번들경로>:/bundle -w /work \\
  ultralytics/ultralytics:latest-jetson-jetpack6
```

## 2. 엔진 빌드 (보드에서, 두 개 다)
🔴 x86 엔진은 안 돈다. **.pt 에서** 빌드한다 — ONNX는 ultralytics로 재빌드가 안 된다.
```bash
python3 deploy/export_trt.py --best /bundle/weights/{w} --imgsz 736,1280 --formats fp16  # 배포용(rect)
python3 deploy/export_trt.py --best /bundle/weights/{w} --imgsz 1280     --formats fp16  # mAP 비교용(정사각)
```
왜 둘 다: ultralytics `val()` 이 rect(imgsz=[h,w])를 지원하지 않아서(§15-6),
**mAP 축은 정사각 엔진**으로, **운영점·속도는 rect 엔진**으로 잰다.

## 3. 정확도 — x86 기준선과 같은 축
```bash
# A) 표준 mAP (DATASET.md §6-2: imgsz 1280, conf 0.001 기본값)  → 기준 군함 mAP50 0.9788
yolo detect val model=engines/{Path(w).stem}_1280_fp16.engine \\
    data=/bundle/marine_board.yaml imgsz=1280 plots=False

# B) 운영점 (배포 설정 그대로)  → 기준 recall@0.6 0.9755 / 오경보 2.12%
python3 deploy/eval_op.py --weights engines/{Path(w).stem}_736x1280_fp16.engine \\
    --imgsz 736x1280 --data-root /bundle/benchmark/marine_session_spot --conf 0.6 \\
    --csv results/deploy/op_jetson.csv
```
합격선: FP16 mAP 손실 ≤1%p · 군함 recall@0.6 ≥0.95.

## 4. 속도 — 보드 숫자는 여기서만 나온다
```bash
python3 deploy/bench_engine.py --weights engines/{Path(w).stem}_736x1280_fp16.engine \\
    --imgsz-list 736x1280 --frames /bundle/benchmark/marine_session_spot/images/{args.split} --n 200

# end-to-end (헤드리스 = 배포 조건, 저장·표시 없음)
python3 deploy/stream_infer.py --source /bundle/demo/{demo} \\
    --model engines/{Path(w).stem}_736x1280_fp16.engine --imgsz 736,1280 \\
    --conf 0.25 --alert-conf 0.6 --n 6 --m 10

# 전력모드별 × 열스로틀 후 지속 성능 (배포 숫자는 '지속' 쪽)
bash deploy/jetson/power_bench.sh engines/{Path(w).stem}_736x1280_fp16.engine 736x1280 \\
    /bundle/benchmark/marine_session_spot/images/{args.split} 0 1 2
```

## 5. 숫자 볼 때 (이거 틀리면 보고가 틀어진다)
- **평균 말고 p50.** 첫 프레임이 TRT 워밍업·의존성 자동설치로 수 초 걸려 평균을 통째로 오염시킨다(§15-6).
- **저장·표시 끄고 잰 값이 배포 FPS.** 오버레이+인코딩만 x86에서 프레임당 ~20ms 였고,
  Orin Nano 는 **하드웨어 인코더(NVENC)가 없어**(디코더는 있음) 더 비싸다 — `setup.sh` 출력으로 확인.
  데모 영상은 FPS 측정과 분리해서 따로 뽑을 것.
- **열스로틀 후 지속 FPS** 가 배포 숫자다. 첫 30초 피크 아님.
- **전처리를 따로 볼 것.** x86에서 이미 전처리 4.7ms > 추론 2.25ms 였다(P1 ⑥).
  30 FPS 미달이면 범인은 모델이 아니라 전처리일 가능성이 높다 → nvvidconv/DeepStream 로 넘기는 게 다음 수.
- **INT8 재검증 대상.** x86 탈락은 TRT 11 + ultralytics 8.4 조합의 툴체인 판정이었다.
  보드는 TRT 10.x 라 결과가 다를 수 있다. 단 **1280 INT8 빌드 전에 640 스모크로 속도부터** 볼 것(빌드 30분+).
"""


if __name__ == "__main__":
    main()
