"""
세션(날짜·지점) 단위 train/val 분할 — leakage 없는 '진짜 일반화' 평가용.

기존 build_split.py는 clip 단위라, 같은 (날짜,지점) 세션이 여러 clip으로 쪼개져
train/val 양쪽에 들어감(= 같은 실제 군함·배경을 외운 걸 다시 채점). audit 결과
val 군함 세션의 83%가 train과 겹쳤음. 이 스크립트는 **세션 통째로** 한쪽에만 배정.

옵션:
  --holdout-spots S1,S2 : 해당 spot을 통째로 val로 (→ '처음 보는 지점' 일반화까지 평가)

출력: splits_session/train_clips.txt, val_clips.txt (+ leakage/겹침 통계)
"""
import argparse, csv, random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="../aug/out_train/clips.csv")
    ap.add_argument("--out", default="splits_session")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-spots", default="", help="쉼표구분 spot들을 통째로 val로")
    args = ap.parse_args()

    rows = []
    with open(args.clips, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    hold = {s.strip() for s in args.holdout_spots.split(",") if s.strip()}

    # 세션 = (date, spot). 세션별 clip 목록·군함여부 집계
    sess = {}
    for r in rows:
        k = (r["date"], r["spot"])
        d = sess.setdefault(k, {"clips": [], "warship": False, "spot": r["spot"]})
        d["clips"].append(r["clip_id"])
        if r["has_warship"] == "1":
            d["warship"] = True

    keys = sorted(sess.keys())
    # holdout spot 세션은 무조건 val
    forced_val = [k for k in keys if sess[k]["spot"] in hold]
    pool = [k for k in keys if sess[k]["spot"] not in hold]

    # 나머지는 군함유무로 층화해 세션 단위 분할
    rng = random.Random(args.seed)
    wpool = sorted(k for k in pool if sess[k]["warship"])
    npool = sorted(k for k in pool if not sess[k]["warship"])
    train_k, val_k = [], list(forced_val)
    for grp in (wpool, npool):
        g = grp[:]; rng.shuffle(g)
        nv = round(len(g) * args.val_frac)
        val_k += g[:nv]; train_k += g[nv:]

    train = sorted(c for k in train_k for c in sess[k]["clips"])
    val = sorted(c for k in val_k for c in sess[k]["clips"])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "train_clips.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "val_clips.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    # --- 검증 통계 ---
    def wspots(kset):
        return {sess[k]["spot"] for k in kset if sess[k]["warship"]}
    tw_s, vw_s = set(train_k), set(val_k)
    tw_spots, vw_spots = wspots(tw_s), wspots(vw_s)
    nframes = {r["clip_id"]: int(r["n_frames"]) for r in rows}

    print(f"seed={args.seed} val_frac={args.val_frac} holdout_spots={sorted(hold) or '없음'}")
    print(f"세션 {len(keys)} → train {len(train_k)} / val {len(val_k)}")
    print(f"clips  train {len(train)} ({sum(nframes[c] for c in train)}f) / "
          f"val {len(val)} ({sum(nframes[c] for c in val)}f)")
    ntw = len([k for k in tw_s if sess[k]["warship"]])
    nvw = len([k for k in vw_s if sess[k]["warship"]])
    print(f"군함 세션  train {ntw} / val {nvw}")
    # 핵심: 세션 leakage(같은 date,spot이 양쪽) = 0 이어야 함(세션 단위라 구조적으로 0)
    print(f"세션 leakage (train∩val 세션): {len(tw_s & vw_s)}  ← 0이어야 정상")
    print(f"군함 spot  train {sorted(tw_spots)} / val {sorted(vw_spots)}")
    print(f"val 전용 군함 spot(처음 보는 지점): {sorted(vw_spots - tw_spots) or '없음'}")


if __name__ == "__main__":
    main()
