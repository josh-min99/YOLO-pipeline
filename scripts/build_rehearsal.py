"""
rehearsal용 합성 야간 서브셋 생성 (W8b).

왜 필요한가: stage2를 실데이터(전부 주간)만으로 돌렸더니 stage1이 익힌 야간
특징이 씻겨나갔다 — 71856 전체 군함 mAP50이 0.876 -> 0.449. 학습 중에 야간
샘플을 계속 보여주면(rehearsal) 그쪽 손실이 살아있어 가중치가 주간 전용으로
치우치지 못한다.

비율: 실데이터 train 28,698장에 맞춰 기본 28,000장(약 1:1). EO 야간과 IR 야간을
같은 수로 뽑는다 — 한쪽만 많으면 그 센서로 치우친다.

출력은 심링크라 디스크를 거의 안 쓴다(§14-6대로 절대경로 심링크).
"""
import argparse, random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/synth_train", help="synth71856_to_yolo.py 산출물")
    ap.add_argument("--out", default="datasets/synth_night")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=28000, help="총 표본 수(센서별로 균등 분할)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_img = Path(args.src) / "images" / args.split
    src_lab = Path(args.src) / "labels" / args.split
    out_img = Path(args.out) / "images" / args.split
    out_lab = Path(args.out) / "labels" / args.split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    # 센서별로 나눠 균등 표집. 파일명 접두가 EO_/IR_ 로 구분된다.
    groups = {"EO": [], "IR": []}
    for p in src_img.glob("*.jpg"):
        groups.setdefault(p.name[:2], []).append(p)
    for k in groups:
        print(f"  원본 {k}: {len(groups[k])}장")

    rng = random.Random(args.seed)
    per = args.n // max(len([k for k in groups if groups[k]]), 1)
    picked = []
    for k, lst in groups.items():
        if not lst:
            continue
        rng.shuffle(lst)
        picked += lst[:per]
    print(f"표집: {len(picked)}장 (센서별 {per})")

    n_link = n_skip = 0
    for ip in picked:
        lp = src_lab / f"{ip.stem}.txt"
        if not lp.exists():
            n_skip += 1
            continue
        for s, d in ((ip, out_img / ip.name), (lp, out_lab / lp.name)):
            if d.is_symlink() and not d.exists():
                d.unlink()                       # 이전 실행이 남긴 깨진 링크 정리
            if not (d.exists() or d.is_symlink()):
                try:
                    d.symlink_to(s.resolve())    # 상대링크는 깨진다(§14-6)
                except (OSError, NotImplementedError):
                    import shutil; shutil.copy2(s, d)
        n_link += 1
    print(f"링크 {n_link}장 (라벨 없어 스킵 {n_skip})")
    print(f"-> {out_img}")


if __name__ == "__main__":
    main()
