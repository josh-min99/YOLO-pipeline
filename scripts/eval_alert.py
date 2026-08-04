"""
N-of-M 시간적 경보 규칙의 효과 측정 — 프레임 지표가 아니라 **이벤트 지표**로.

왜 필요한가: 배포에서 중요한 건 프레임 정확도가 아니라 경보의 품질이다.
프레임 단위 오탐이 2.12%(새 장소, conf 0.6)면 30 FPS에서 초당 0.6건이라 실사용이
안 된다. 그러나 진짜 표적은 연속 프레임에 걸쳐 잡히고 오탐은 단발이므로,
"최근 M프레임 중 N프레임 이상"이라는 규칙이 단발만 골라 죽일 수 있다.
그것이 실제로 되는지를 숫자로 확인한다(deploy/alert.py의 규칙과 동일).

측정 방식
---------
1) 추론은 클립별로 프레임 순서대로 **한 번만** 돌려 프레임별 '군함 최고 conf'를 저장.
2) conf/N/M은 그 배열 위에서 후처리로 훑는다 — 임계값 스윕이 공짜가 된다.

지표
----
- 군함 클립: 경보가 떴는가(이벤트 recall), 몇 프레임 만에 떴는가(지연)
- 군함 없는 클립: 헛경보가 났는가(이벤트 오경보율) — 운영에서 실제로 비용이 되는 값

usage:
  python scripts/eval_alert.py --weights best_spot.pt --data-root datasets/marine \
      --conf-list 0.10,0.25,0.40,0.60 --n 6 --m 10
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def clip_of(stem):
    """I1_S0_C5_0026003 -> I1_S0_C5_0026 (마지막 3자리가 프레임 번호)."""
    return stem[:-3]


def frame_idx(stem):
    return int(stem[-3:])


def scan_frames(root, split, warship_class=2):
    """클립별 (프레임순) stem 목록과 '군함 있음' 여부."""
    lab_dir = Path(root) / "labels" / split
    wc = str(warship_class)
    clips = defaultdict(list)
    has_w = defaultdict(bool)
    for lab in lab_dir.glob("*.txt"):
        stem = lab.stem
        clips[clip_of(stem)].append(stem)
        for ln in lab.read_text(encoding="utf-8").splitlines():
            if ln.split()[:1] == [wc]:
                has_w[clip_of(stem)] = True
                break
    for c in clips:
        clips[c].sort(key=frame_idx)
    return clips, has_w


def alert_events(scores, conf, n, m, off_patience):
    """
    프레임별 군함 conf 배열 -> 경보 ON 구간 목록.
    deploy/alert.py 와 같은 규칙: 최근 m 중 n 이상 히트하면 ON,
    이후 off_patience 연속 미검출이면 OFF.
    반환: [(on_index, ...)] 의 on 시작 인덱스 목록
    """
    on = False
    miss = 0
    starts = []
    for i in range(len(scores)):
        window = scores[max(0, i - m + 1): i + 1]
        hits = sum(1 for s in window if s >= conf)
        if not on:
            if hits >= n:
                on = True
                miss = 0
                starts.append(i)
        else:
            if scores[i] >= conf:
                miss = 0
            else:
                miss += 1
                if miss >= off_patience:
                    on = False
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default="datasets/marine")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf-list", default="0.10,0.25,0.40,0.60")
    ap.add_argument("--n", type=int, default=6, help="M 중 N 프레임 이상이면 경보 ON")
    ap.add_argument("--m", type=int, default=10, help="판정 윈도")
    ap.add_argument("--off-patience", type=int, default=15)
    ap.add_argument("--warship-class", type=int, default=2)
    ap.add_argument("--cache", default="results/alert_scores.json",
                    help="프레임별 군함 conf 캐시(있으면 추론 생략)")
    ap.add_argument("--out", default="results/alert_sweep.csv")
    args = ap.parse_args()

    root = Path(args.data_root)
    clips, has_w = scan_frames(root, args.split, args.warship_class)
    print(f"클립 {len(clips)}개 (군함 {sum(has_w.values())} / 없음 {len(clips)-sum(has_w.values())})")

    cache = Path(args.cache)
    if cache.exists():
        scores = json.loads(cache.read_text(encoding="utf-8"))
        print(f"캐시 사용: {cache}")
    else:
        from ultralytics import YOLO
        model = YOLO(args.weights)
        img_dir = root / "images" / args.split
        scores = {}
        for ci, (c, stems) in enumerate(sorted(clips.items()), 1):
            paths = [str(img_dir / f"{s}.jpg") for s in stems]
            vals = []
            # 클립 단위 배치 추론. conf는 최저로 두고 임계값은 후처리에서 건다.
            for i in range(0, len(paths), 32):
                for r in model.predict(paths[i:i + 32], imgsz=args.imgsz, conf=0.001,
                                       device=args.device, verbose=False):
                    w = [float(b.conf) for b in r.boxes if int(b.cls) == args.warship_class]
                    vals.append(max(w) if w else 0.0)
            scores[c] = vals
            if ci % 30 == 0:
                print(f"  {ci}/{len(clips)} 클립")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(scores), encoding="utf-8")
        print(f"캐시 저장: {cache}")

    rows = []
    print(f"\n규칙: 최근 {args.m}프레임 중 {args.n}프레임 이상 → 경보 ON")
    print(f"{'conf':>6}{'프레임오탐%':>12}{'이벤트오경보%':>14}{'이벤트recall%':>14}{'지연(프레임)':>13}")
    for conf in [float(x) for x in args.conf_list.split(",")]:
        n_pos = n_pos_alert = 0
        n_neg = n_neg_alert = 0
        frame_fp = frame_neg = 0
        delays = []
        for c, stems in clips.items():
            s = scores.get(c, [])
            if not s:
                continue
            ev = alert_events(s, conf, args.n, args.m, args.off_patience)
            if has_w[c]:
                n_pos += 1
                if ev:
                    n_pos_alert += 1
                    delays.append(ev[0])
            else:
                n_neg += 1
                if ev:
                    n_neg_alert += 1
                frame_neg += len(s)
                frame_fp += sum(1 for v in s if v >= conf)
        med = sorted(delays)[len(delays) // 2] if delays else float("nan")
        fpr = 100 * frame_fp / max(frame_neg, 1)
        evfp = 100 * n_neg_alert / max(n_neg, 1)
        evrec = 100 * n_pos_alert / max(n_pos, 1)
        print(f"{conf:>6.2f}{fpr:>12.2f}{evfp:>14.2f}{evrec:>14.1f}{med:>13.0f}")
        rows.append(dict(conf=conf, frame_fp_pct=round(fpr, 3), event_fa_pct=round(evfp, 2),
                         event_recall_pct=round(evrec, 2), median_delay_frames=med,
                         n_pos_clips=n_pos, n_neg_clips=n_neg))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n저장: {out}")
    print("※ 프레임오탐% = 군함 없는 클립의 프레임 중 군함이 검출된 비율 (기존 지표)")
    print("  이벤트오경보% = 군함 없는 클립 중 경보가 한 번이라도 뜬 클립 비율 (운영 지표)")


if __name__ == "__main__":
    main()
