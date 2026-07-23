"""
clip 단위 train/val 분할 생성.

per-frame 탐지기라 세그먼트(연속구간) 제약이 필요 없다 → 전 프레임이 유효 샘플.
누수 방지를 위해 **clip 단위**로 나누고, has_warship로 층화(군함 클립이 train/val 양쪽에).
재구성 VAD의 분할(군함 클립을 학습서 제외)과 반대 — 탐지기는 군함을 학습서 봐야 함.

입력:  aug/out_train/clips.csv (863 클립)
출력:  splits/train_clips.txt, splits/val_clips.txt  (+ stdout 통계)
"""
import argparse, csv, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="../aug/out_train/clips.csv")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    with open(args.clips, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # 층화: 군함 유무로 두 그룹 나눠 각각 val_frac만큼 val로
    warship = sorted(r["clip_id"] for r in rows if r["has_warship"] == "1")
    normal  = sorted(r["clip_id"] for r in rows if r["has_warship"] != "1")
    nframes = {r["clip_id"]: int(r["n_frames"]) for r in rows}

    rng = random.Random(args.seed)
    train, val = [], []
    for group in (warship, normal):
        g = group[:]
        rng.shuffle(g)
        n_val = round(len(g) * args.val_frac)
        val.extend(g[:n_val])
        train.extend(g[n_val:])
    train.sort(); val.sort()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "train_clips.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "val_clips.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    def stat(name, clips):
        nw = sum(1 for c in clips if c in warship)
        nf = sum(nframes[c] for c in clips)
        print(f"  {name:5s}: {len(clips):3d} clips ({nw:3d} warship) / {nf:6d} frames")

    print(f"seed={args.seed} val_frac={args.val_frac}")
    print(f"total: {len(rows)} clips ({len(warship)} warship / {len(normal)} normal)")
    stat("train", train); stat("val", val)

if __name__ == "__main__":
    main()
