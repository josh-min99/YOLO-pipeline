"""
군함 오탐(false alarm) 측정 — 배포에서 mAP보다 중요한 운영 지표.

'군함이 없는' val 프레임(라벨에 class 2 없음)에 탐지기를 돌려서,
conf 임계값별로 '가짜 군함 경보'가 얼마나 나는지 센다. audit의 핵심 우려
(빈 바다/정상 선박에서의 오탐)를 직접 수치화.

사용(vast, best.pt 있는 곳):
  python scripts/eval_warship_fp.py --best <best.pt> --data-root datasets/marine \
      --conf-list 0.25,0.4,0.6 --split val
"""
import argparse
from pathlib import Path


def collect_negatives(data_root, split="val", warship_class=2):
    """군함이 없는(라벨에 해당 클래스 없음) 프레임 이미지 경로 목록. deploy/eval_engine.py 에서도 사용."""
    root = Path(data_root)
    lab_dir, img_dir = root / "labels" / split, root / "images" / split
    wc = str(warship_class)
    negs = []
    for lab in lab_dir.glob("*.txt"):
        has_w = any(ln.split()[:1] == [wc] for ln in lab.read_text().splitlines() if ln.strip())
        if not has_w:
            img = img_dir / (lab.stem + ".jpg")
            if img.exists():
                negs.append(str(img))
    return negs


def count_false_alarms(model, negs, conf, warship_class=2, imgsz=1280, batch=16):
    """(가짜경보 프레임수, 가짜 군함 박스수) — negs 에는 실제 군함이 0이므로 검출=전부 오탐."""
    frames_fp = boxes_fp = 0
    for i in range(0, len(negs), batch):
        res = model.predict(negs[i:i + batch], conf=conf, classes=[warship_class],
                            imgsz=imgsz, verbose=False)
        for r in res:
            n = len(r.boxes)
            if n:
                frames_fp += 1
                boxes_fp += n
    return frames_fp, boxes_fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True)
    ap.add_argument("--data-root", default="datasets/marine")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf-list", default="0.25,0.4,0.6")
    ap.add_argument("--warship-class", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    negs = collect_negatives(args.data_root, args.split, args.warship_class)
    print(f"군함 없는 {args.split} 프레임: {len(negs)}")
    if not negs:
        raise SystemExit("군함 없는 프레임이 없음 — data-root/split 확인")

    from ultralytics import YOLO
    model = YOLO(args.best)

    print(f"\n{'conf':>6} {'가짜경보 프레임':>14} {'프레임오탐율':>12} {'가짜군함 박스수':>14}")
    for conf in [float(c) for c in args.conf_list.split(",")]:
        # 군함 클래스만 예측(classes=[2])해서 빠르게
        frames_fp, boxes_fp = count_false_alarms(model, negs, conf, args.warship_class,
                                                 args.imgsz, args.batch)
        rate = frames_fp / len(negs)
        print(f"{conf:>6.2f} {frames_fp:>14} {rate:>11.1%} {boxes_fp:>14}")

    print("\n해석: 이 프레임들엔 실제 군함이 0이므로, 예측된 군함은 전부 가짜 경보.")
    print("배포 임계값(F1최적≈0.6) 기준 프레임오탐율이 실사용 오경보율의 근사치.")


if __name__ == "__main__":
    main()
