"""
YOLO 데이터셋 -> VLM 학습용 JSONL 변환 (YOLO vs VLM 정면 비교용).

라벨 JSON에서 새로 만들지 않고 **이미 빌드된 YOLO 데이터셋 디렉토리를 그대로 읽는다.**
이유: 분할(train/val)이 YOLO run과 100% 동일함을 보장하기 위함. 여기서 재빌드하면
split 불일치가 조용히 섞여 비교가 무효가 된다.

입력 (ultralytics 표준 레이아웃, scripts/json_to_yolo.py 산출물):
    <root>/images/<split>/<stem>.jpg
    <root>/labels/<split>/<stem>.txt    (cls cx cy w h, 0~1 정규화)

출력 (모델 비종속 중간형식):
    <out>/<split>.jsonl
    {"image": "<절대경로>", "width": 1920, "height": 1080,
     "objects": [{"bbox": [x1,y1,x2,y2], "label": "warship"}]}   # 원본 픽셀 xyxy

좌표는 **원본 이미지 픽셀 공간**의 xyxy로 둔다. 모델별 좌표 규약(Qwen의 smart_resize
이후 공간, Florence-2의 <loc> 1024 bin 등)은 학습 스크립트의 formatter에서 변환한다.
여기서 미리 모델 좌표로 굽지 말 것 -- 모델 갈아탈 때마다 데이터를 다시 만들게 된다.

주요 옵션:
    --stride N       클립 내 N프레임마다 1장만 사용(연속 프레임은 거의 중복).
                     파일럿 run으로 비용을 줄일 때. 헤드라인 숫자는 stride=1로.
    --check-boxes K  K장을 골라 박스를 그려 저장. 좌표 규약 검증용(§16-6 예방).

usage:
    python scripts/vlm_prepare.py --root datasets/marine_session_spot --out datasets/marine_vlm
    python scripts/vlm_prepare.py --root datasets/marine_session_spot --out datasets/marine_vlm \
        --stride 4 --check-boxes 12
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

# configs/marine.yaml 과 반드시 동일한 순서. 여기가 어긋나면 클래스가 통째로 섞인다.
NAMES = ["fishing_boat", "merchant_ship", "warship"]


def clip_of(stem):
    """I1_S0_C5_0026003 -> I1_S0_C5_0026 (마지막 3자리가 프레임 번호, §10-3)."""
    return stem[:-3]


def read_label(txt_path, W, H):
    """YOLO txt -> [{'bbox': [x1,y1,x2,y2], 'label': name}]. 없거나 비면 빈 리스트."""
    if not txt_path.exists():
        return []
    objs = []
    for ln in txt_path.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:])
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        # 원본 라벨이 프레임 밖으로 살짝 나가는 경우 방어
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2, y2 = min(x2, float(W)), min(y2, float(H))
        if x2 <= x1 or y2 <= y1:
            continue
        objs.append({"bbox": [round(v, 1) for v in (x1, y1, x2, y2)], "label": NAMES[c]})
    return objs


def image_size(p):
    """PIL로 헤더만 읽어 크기 취득(전체 디코드 안 함). 한글 경로 안전(§9-4)."""
    from PIL import Image

    with Image.open(p) as im:
        return im.size  # (W, H)


def draw_check(records, out_dir, k, seed=0):
    """
    좌표 규약 검증: GT를 그려서 실제로 배 위에 얹히는지 눈으로 본다.
    §16-6에서 [w,h,x,y]를 [x,y,w,h]로 읽고도 경계검사를 통과했던 사고의 예방책.
    경계/포맷 검사는 대칭이라 증거가 못 되고, 그림만이 증거다.
    """
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # 객체가 있는 것 중에서, 가급적 작은 박스가 포함된 샘플을 고른다
    cand = [r for r in records if r["objects"]]
    picks = rng.sample(cand, min(k, len(cand)))
    for r in picks:
        im = Image.open(r["image"]).convert("RGB")
        dr = ImageDraw.Draw(im)
        for o in r["objects"]:
            x1, y1, x2, y2 = o["bbox"]
            dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            dr.text((x1, max(0, y1 - 12)), o["label"], fill=(255, 0, 0))
        im.save(out_dir / (Path(r["image"]).stem + ".jpg"), quality=88)
    print(f"[check] {len(picks)} images -> {out_dir}")
    print("  >> 박스가 배 위에 정확히 얹혔는지 반드시 눈으로 확인할 것.")


def build_split(root, split, stride):
    img_dir = root / "images" / split
    lab_dir = root / "labels" / split
    if not img_dir.is_dir():
        raise SystemExit(f"없는 경로: {img_dir}")

    stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if stride > 1:
        # 클립 단위로 stride 적용(전역 stride는 클립 경계에서 편향된다)
        by_clip = {}
        for s in stems:
            by_clip.setdefault(clip_of(s), []).append(s)
        stems = sorted(s for v in by_clip.values() for s in v[::stride])

    records, n_empty, cls_count = [], 0, Counter()
    for s in stems:
        ip = (img_dir / f"{s}.jpg").resolve()
        W, H = image_size(ip)
        objs = read_label(lab_dir / f"{s}.txt", W, H)
        if not objs:
            n_empty += 1
        for o in objs:
            cls_count[o["label"]] += 1
        records.append({"image": str(ip), "width": W, "height": H, "objects": objs})
    return records, n_empty, cls_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="YOLO 데이터셋 루트(images/, labels/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--stride", type=int, default=1,
                    help="클립 내 N프레임마다 1장. 파일럿용. 헤드라인은 1로 둘 것")
    ap.add_argument("--check-boxes", type=int, default=0,
                    help="K장에 GT를 그려 저장(좌표 규약 검증)")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(","):
        split = split.strip()
        recs, n_empty, cls_count = build_split(root, split, args.stride)
        jp = out / f"{split}.jsonl"
        with open(jp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        n_obj = sum(cls_count.values())
        print(f"[{split}] {len(recs)} images / {n_obj} boxes -> {jp}")
        for name in NAMES:
            print(f"    {name:14s} {cls_count[name]}")
        print(f"    empty(객체 0개) {n_empty}  ({n_empty / max(len(recs), 1):.1%})")
        if n_empty == 0:
            print("    🔴 빈 프레임 0개: VLM이 '없음'이라고 답하는 법을 못 배운다.")
            print("       -> 빈 바다에서 전면 환각 위험. 결과 보고 시 이 한계를 명시할 것.")

        if args.check_boxes:
            draw_check(recs, out / f"check_{split}", args.check_boxes)


if __name__ == "__main__":
    main()
