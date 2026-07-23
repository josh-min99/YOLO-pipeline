"""
AI Hub 이미지 zip(TS_*.zip) -> flat stem.jpg 추출.

탐지기는 per-frame이라 세그먼트/재번호(s2_build_frames) 불필요 → 전 프레임을 그냥 평면으로 푼다.
zip 항목은 flat(`/I1_S0_C5_0001011.jpg`), ~1% 손상(BadZipFile) → 손상 항목만 건너뜀(§10-3).

사용(vast):
    python scripts/extract_images.py --zip-dir <TS_zip들 있는 폴더> --out ../datasets/marine_frames
"""
import argparse, zipfile
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", required=True, help="TS_*.zip 들이 있는 폴더")
    ap.add_argument("--out", required=True, help="stem.jpg 들을 풀 폴더")
    ap.add_argument("--pattern", default="TS_*.zip")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    zips = sorted(Path(args.zip_dir).glob(args.pattern))
    if not zips:
        raise SystemExit(f"zip 못 찾음: {args.zip_dir}/{args.pattern}")

    n_ok = n_bad = 0
    bad_examples = []
    for zp in zips:
        print(f"opening {zp.name} ...", flush=True)
        zf = zipfile.ZipFile(zp)
        for info in zf.infolist():
            name = info.filename
            if not name.lower().endswith(".jpg"):
                continue
            stem = Path(name).stem            # 선행 '/' 및 경로 제거
            dst = out / f"{stem}.jpg"
            if dst.exists():
                n_ok += 1
                continue
            try:
                dst.write_bytes(zf.read(info))  # 손상 항목은 여기서 예외
                n_ok += 1
            except Exception as ex:
                n_bad += 1
                if len(bad_examples) < 10:
                    bad_examples.append(f"{stem}: {type(ex).__name__}")
            if n_ok % 5000 == 0 and n_ok:
                print(f"  ...{n_ok} frames", flush=True)

    print(f"\n[done] extracted {n_ok} frames / skipped {n_bad} corrupt")
    print(f"  out: {out}")
    for b in bad_examples:
        print(f"    bad: {b}")

if __name__ == "__main__":
    main()
