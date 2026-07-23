"""
AI Hub 라벨 JSON -> YOLO 데이터셋 변환.

이미지가 있는 곳(vast 또는 로컬 서브셋)에서 실행. 라벨 JSON은 stem별 1개.
clip = stem[:-3] (예: I1_S0_C5_0001001 -> I1_S0_C5_0001) 로 split에 매칭.

클래스(군함 우선, 사람/조류 제외):
    "0" 어선 -> 0,  "1" 상선 -> 1,  "2" 군함 -> 2   (그 외 클래스의 bbox는 스킵)
어노테이션 없는 프레임 -> 빈 .txt (배경 negative, 탐지기에 유익).

출력 레이아웃 (ultralytics 표준):
    <out>/images/<split>/<stem>.jpg
    <out>/labels/<split>/<stem>.txt   (cls cx cy w h, 전부 0~1 정규화)

--labels-only : 이미지 없이 .txt만 생성(로컬 검증용).
"""
import argparse, csv, json, shutil
from pathlib import Path

CLASS_MAP = {"0": 0, "1": 1, "2": 2}  # 어선/상선/군함. 사람"3"/조류"4" 제외.

def read_clip_list(p):
    return [ln.strip() for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]

def convert_one(jpath, W_default=1920, H_default=1080):
    """label JSON -> list of 'cls cx cy w h' 문자열. 빈 리스트면 배경 프레임."""
    d = json.loads(Path(jpath).read_text(encoding="utf-8"))
    lines, skipped = [], 0
    for a in d.get("annotations", []):
        c = CLASS_MAP.get(str(a.get("class")))
        if c is None:
            skipped += 1
            continue
        W = a.get("width", W_default) or W_default
        H = a.get("height", H_default) or H_default
        x, y, w, h = a["bbox"]
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        nw, nh = w / W, h / H
        # 경계 클램프(라벨이 프레임 밖으로 살짝 나가는 경우 방어)
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        nw, nh = min(nw, 1), min(nh, 1)
        lines.append(f"{c} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines, skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True, help="stem별 라벨 JSON 폴더")
    ap.add_argument("--images-dir", help="stem별 이미지 폴더(재귀 탐색). --labels-only면 불필요")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-ext", default=".jpg")
    ap.add_argument("--link", action="store_true", help="복사 대신 심볼릭 링크(디스크 절약)")
    ap.add_argument("--labels-only", action="store_true")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    out = Path(args.out)

    # 이미지 인덱스 1회 구축(stem -> path). rglob을 프레임마다 돌리지 말 것(§9-5).
    img_index = {}
    if not args.labels_only:
        assert args.images_dir, "--images-dir 필요(또는 --labels-only)"
        for p in Path(args.images_dir).rglob(f"*{args.img_ext}"):
            img_index[p.stem] = p
        print(f"image index: {len(img_index)} files")

    for split in ("train", "val"):
        clips = set(read_clip_list(Path(args.splits_dir) / f"{split}_clips.txt"))
        img_out = out / "images" / split
        lab_out = out / "labels" / split
        lab_out.mkdir(parents=True, exist_ok=True)
        if not args.labels_only:
            img_out.mkdir(parents=True, exist_ok=True)

        n_frame = n_box = n_bg = n_miss = 0
        for jpath in labels_dir.glob("*.json"):
            stem = jpath.stem
            if stem[:-3] not in clips:  # clip 필터
                continue
            if not args.labels_only:
                ip = img_index.get(stem)
                if ip is None:
                    n_miss += 1
                    continue
                dst = img_out / f"{stem}{args.img_ext}"
                if not dst.exists():
                    if args.link:
                        try: dst.symlink_to(ip)
                        except (OSError, NotImplementedError): shutil.copy2(ip, dst)
                    else:
                        shutil.copy2(ip, dst)
            lines, _ = convert_one(jpath)
            (lab_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_frame += 1; n_box += len(lines)
            if not lines: n_bg += 1
        print(f"[{split}] frames={n_frame} boxes={n_box} background={n_bg} img_missing={n_miss}")

if __name__ == "__main__":
    main()
