"""
YOLO vs VLM 공정 비교 평가기.

지표 코드는 **하나만** 둔다. YOLO 예측과 VLM 예측을 같은 포맷으로 뽑아 같은 함수에
넣어야 "ultralytics mAP vs 내가 짠 mAP"라는 비교 불가 상황을 피한다.

  predict --backend yolo --weights best_spot.pt --data datasets/marine_vlm/val.jsonl --out preds_yolo.json
  predict --backend vlm  --adapter runs/vlm/qwen25vl3b/adapter_final --data ... --out preds_vlm.json
  score preds_yolo.json preds_vlm.json --conf 0.6

예측 포맷(공통):
  {"meta": {...}, "preds": {"<image path>": [{"bbox":[x1,y1,x2,y2],"label":str,"score":float}, ...]}}
  bbox는 항상 **원본 이미지 픽셀 xyxy**. VLM은 여기서 resize 공간을 되돌려 담는다.

🔴 VLM은 confidence를 안 뱉는다. mAP는 랭킹이 있어야 정의되므로, 생성 토큰의
   logprob 평균을 점수로 쓴다(--score-mode logprob). 그래도 YOLO의 conf만큼
   잘 교정된 값이 아니므로 **헤드라인은 운영점 P/R/F1로 보고**하고 mAP는 참고로 둘 것.
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

NAMES = ["fishing_boat", "merchant_ship", "warship"]


# ---------------------------------------------------------------- 지표

def iou_matrix(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def ap_for_class(preds, gts, cls, thr=0.5):
    """
    preds: {img: [(bbox, label, score)]}, gts: {img: [(bbox, label)]}
    all-point interpolation AP. 반환 (ap, n_gt).
    """
    rows = []
    for img, ds in preds.items():
        for bb, lb, sc in ds:
            if lb == cls:
                rows.append((sc, img, bb))
    rows.sort(key=lambda r: -r[0])

    gt_by_img = {img: [g[0] for g in gs if g[1] == cls] for img, gs in gts.items()}
    n_gt = sum(len(v) for v in gt_by_img.values())
    if n_gt == 0:
        return float("nan"), 0
    used = {img: np.zeros(len(v), bool) for img, v in gt_by_img.items()}

    tp = np.zeros(len(rows))
    fp = np.zeros(len(rows))
    for i, (_, img, bb) in enumerate(rows):
        g = gt_by_img.get(img, [])
        if not g:
            fp[i] = 1
            continue
        ious = iou_matrix([bb], g)[0]
        j = int(np.argmax(ious))
        if ious[j] >= thr and not used[img][j]:
            used[img][j] = True
            tp[i] = 1
        else:
            fp[i] = 1

    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # precision envelope
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), n_gt


def operating_point(preds, gts, cls, conf, thr=0.5):
    """conf 이상만 남기고 P/R/F1. 배포 판단은 이 숫자로 한다."""
    tp = fp = 0
    n_gt = sum(sum(1 for g in gs if g[1] == cls) for gs in gts.values())
    for img in gts:
        g = [x[0] for x in gts[img] if x[1] == cls]
        d = sorted([p for p in preds.get(img, []) if p[1] == cls and p[2] >= conf], key=lambda p: -p[2])
        used = np.zeros(len(g), bool)
        for bb, _, _ in d:
            if not g:
                fp += 1
                continue
            ious = iou_matrix([bb], g)[0]
            j = int(np.argmax(ious))
            if ious[j] >= thr and not used[j]:
                used[j] = True
                tp += 1
            else:
                fp += 1
    p = tp / max(tp + fp, 1)
    r = tp / max(n_gt, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return dict(P=p, R=r, F1=f1, TP=tp, FP=fp, n_gt=n_gt)


def load_gt(jsonl):
    gts = {}
    for ln in Path(jsonl).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        gts[r["image"]] = [(o["bbox"], o["label"]) for o in r["objects"]]
    return gts


# ---------------------------------------------------------------- 예측: YOLO

def predict_yolo(args, recs):
    from ultralytics import YOLO
    m = YOLO(args.weights)
    out, lat = {}, []
    imgsz = args.imgsz if len(args.imgsz) > 1 else args.imgsz[0]
    for i, r in enumerate(recs):
        t = time.perf_counter()
        res = m.predict(r["image"], imgsz=imgsz, conf=args.min_conf, verbose=False)[0]
        lat.append((time.perf_counter() - t) * 1000)
        dets = []
        for b in res.boxes:
            dets.append({
                "bbox": [float(v) for v in b.xyxy[0].tolist()],
                "label": NAMES[int(b.cls)],
                "score": float(b.conf),
            })
        out[r["image"]] = dets
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(recs)}")
    return out, {"latency_ms_p50": float(np.percentile(lat[args.warmup:], 50)), "parse_fail": 0}


# ---------------------------------------------------------------- 예측: VLM

OBJ_RE = re.compile(r"\{[^{}]*\}")


def predict_vlm(args, recs):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    from PIL import Image
    from vlm_train_qwen import PROMPT, smart_resize

    proc = AutoProcessor.from_pretrained(args.adapter, min_pixels=args.min_pixels,
                                         max_pixels=args.max_pixels)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    out, lat, n_fail = {}, [], 0
    for i, r in enumerate(recs):
        img = Image.open(r["image"]).convert("RGB")
        inputs = proc(text=[prompt_text], images=[img], return_tensors="pt").to("cuda")
        t = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)

        new_ids = gen.sequences[0][inputs["input_ids"].shape[1]:]
        # 토큰별 logprob (greedy이므로 선택 토큰의 확률)
        lps = []
        for step, sc in enumerate(gen.scores):
            if step >= len(new_ids):
                break
            lps.append(float(torch.log_softmax(sc[0].float(), -1)[new_ids[step]]))
        pieces = [proc.tokenizer.decode([t_]) for t_ in new_ids]
        text = "".join(pieces)
        # char offset -> token index 매핑
        starts, acc = [], 0
        for p in pieces:
            starts.append(acc)
            acc += len(p)

        nh, nw = smart_resize(r["height"], r["width"],
                              min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        sx, sy = r["width"] / nw, r["height"] / nh

        dets, ok = [], False
        for m_ in OBJ_RE.finditer(text):
            try:
                o = json.loads(m_.group())
                bb = [float(v) for v in o["bbox_2d"]]
                lb = str(o["label"])
            except Exception:
                continue
            if lb not in NAMES or len(bb) != 4:
                continue
            ok = True
            if args.score_mode == "logprob":
                a, b = m_.start(), m_.end()
                sel = [lps[k] for k, s in enumerate(starts) if a <= s < b and k < len(lps)]
                score = float(np.exp(np.mean(sel))) if sel else 1.0
            else:
                score = 1.0
            # resize 공간 -> 원본 공간
            x1, y1, x2, y2 = bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, r["width"]), min(y2, r["height"])
            if x2 > x1 and y2 > y1:
                dets.append({"bbox": [x1, y1, x2, y2], "label": lb, "score": score})
        # "[]"(객체 없음)은 정상 응답이지 파싱 실패가 아니다
        if not ok and text.strip().replace(" ", "") not in ("[]", "[]<|im_end|>"):
            n_fail += 1
            if n_fail <= 5:
                print(f"  [parse-fail] {Path(r['image']).name}: {text[:120]!r}")
        out[r["image"]] = dets
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(recs)}  parse_fail={n_fail}")

    return out, {
        "latency_ms_p50": float(np.percentile(lat[args.warmup:], 50)),
        "parse_fail": n_fail,
        "parse_fail_rate": n_fail / max(len(recs), 1),
    }


# ---------------------------------------------------------------- CLI

def cmd_predict(args):
    recs = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    print(f"{args.backend}: {len(recs)} images")
    preds, meta = (predict_yolo if args.backend == "yolo" else predict_vlm)(args, recs)
    meta.update(backend=args.backend, data=args.data, n=len(recs))
    Path(args.out).write_text(json.dumps({"meta": meta, "preds": preds}), encoding="utf-8")
    print("saved:", args.out, meta)


def cmd_score(args):
    gts = load_gt(args.data)
    rows = []
    for f in args.preds:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        preds = {k: [(v["bbox"], v["label"], v["score"]) for v in vs] for k, vs in d["preds"].items()}
        gsub = {k: gts[k] for k in preds if k in gts}
        r = {"file": Path(f).name, "meta": d["meta"]}
        for c in NAMES:
            ap, n = ap_for_class(preds, gsub, c)
            r[f"mAP50_{c}"] = ap
        r["mAP50_all"] = float(np.nanmean([r[f"mAP50_{c}"] for c in NAMES]))
        r["op_warship"] = operating_point(preds, gsub, "warship", args.conf)
        rows.append(r)

    print(f"\n{'':28s}" + "".join(f"{Path(f).stem:>22s}" for f in args.preds))
    def line(label, get):
        print(f"{label:28s}" + "".join(f"{get(r):>22s}" for r in rows))
    for c in NAMES:
        line(f"mAP50 {c}", lambda r, c=c: f"{r[f'mAP50_{c}']:.4f}")
    line("mAP50 all", lambda r: f"{r['mAP50_all']:.4f}")
    line(f"warship P @conf{args.conf}", lambda r: f"{r['op_warship']['P']:.4f}")
    line(f"warship R @conf{args.conf}", lambda r: f"{r['op_warship']['R']:.4f}")
    line(f"warship F1 @conf{args.conf}", lambda r: f"{r['op_warship']['F1']:.4f}")
    line("latency ms (p50)", lambda r: f"{r['meta'].get('latency_ms_p50', float('nan')):.1f}")
    line("parse fail rate", lambda r: f"{r['meta'].get('parse_fail_rate', 0):.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("\nsaved:", args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--backend", choices=["yolo", "vlm"], required=True)
    p.add_argument("--data", required=True, help="val.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5, help="§15-6: 첫 프레임이 통계를 오염시킨다")
    # yolo
    p.add_argument("--weights", default="best_spot.pt")
    p.add_argument("--imgsz", type=int, nargs="+", default=[736, 1280])
    p.add_argument("--min-conf", type=float, default=0.001, help="mAP용으로 낮게. 운영점은 score에서 자른다")
    # vlm
    p.add_argument("--adapter", default="runs/vlm/qwen25vl3b/adapter_final")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--max-pixels", type=int, default=1280 * 736)
    p.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--score-mode", choices=["logprob", "const"], default="logprob")
    p.set_defaults(func=cmd_predict)

    s = sub.add_parser("score")
    s.add_argument("preds", nargs="+")
    s.add_argument("--data", required=True, help="val.jsonl (GT)")
    s.add_argument("--conf", type=float, default=0.6, help="W5 운영점")
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
