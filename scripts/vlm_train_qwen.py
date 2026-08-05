"""
Qwen2.5-VL LoRA fine-tune (탐지). YOLO11s와 정면 비교용.

핵심 설계
---------
1) 해상도 패리티. YOLO는 1280x736 rect로 학습했다. VLM에 640 정사각을 넣고 지면
   그건 VLM이 진 게 아니라 해상도가 진 것이다. Qwen2.5-VL은 동적 해상도라
   max_pixels 로 긴 변 ~1280을 종횡비 유지로 맞출 수 있다(기본 1280*736).
2) 좌표 규약. Qwen2.5-VL의 grounding 좌표는 **smart_resize 이후 입력 픽셀 공간**이다.
   원본 1920 좌표를 그대로 학습시키면 박스가 통째로 어긋나 mAP가 0 근처로 나오고
   "VLM은 소형 객체를 못 한다"로 오독하게 된다(§16-6과 같은 계열의 사고).
   -> 여기서 GT를 resize 공간으로 스케일해 넣고, eval에서 되돌린다.
   -> 학습 전 --dump-sample 로 실제 들어가는 문자열을 반드시 눈으로 확인할 것.
3) ViT는 얼리고 LLM에만 LoRA. 3B + bs1 + grad ckpt 로 3090 24GB에 들어간다.

usage (vast, source /venv/main/bin/activate):
    python scripts/vlm_train_qwen.py --data datasets/marine_vlm --out runs/vlm/qwen25vl3b \
        --model Qwen/Qwen2.5-VL-3B-Instruct --epochs 2 --max-pixels 942080
    # 먼저 반드시:
    python scripts/vlm_train_qwen.py --data datasets/marine_vlm --dump-sample 3
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

NAMES = ["fishing_boat", "merchant_ship", "warship"]
PROMPT = (
    "Detect every vessel in this maritime surveillance image. "
    "Return a JSON list; each element has 'bbox_2d' as [x1,y1,x2,y2] in pixels and "
    f"'label' from {NAMES}. If there is no vessel, return []."
)

IMAGE_FACTOR = 28  # patch 14 * merge 2


def smart_resize(h, w, factor=IMAGE_FACTOR, min_pixels=256 * 28 * 28, max_pixels=1280 * 736):
    """
    Qwen2.5-VL의 이미지 리사이즈 규칙 재구현.
    h,w를 factor의 배수로 반올림하고 총 픽셀을 [min,max]에 넣는다. 종횡비 유지.
    processor 내부와 동일한 값을 내야 좌표가 맞는다.
    """
    if max(h, w) / min(h, w) > 200:
        raise ValueError("aspect ratio too extreme")
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def target_text(rec, max_pixels, min_pixels):
    """원본 xyxy -> resize 공간 xyxy -> Qwen 네이티브 grounding JSON 문자열."""
    W, H = rec["width"], rec["height"]
    nh, nw = smart_resize(H, W, min_pixels=min_pixels, max_pixels=max_pixels)
    sx, sy = nw / W, nh / H
    objs = []
    for o in rec["objects"]:
        x1, y1, x2, y2 = o["bbox"]
        objs.append({
            "bbox_2d": [round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)],
            "label": o["label"],
        })
    return json.dumps(objs, ensure_ascii=False)


class DetDataset(Dataset):
    def __init__(self, jsonl, processor, max_pixels, min_pixels, shuffle_seed=None):
        self.recs = [json.loads(ln) for ln in Path(jsonl).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.recs)
        self.proc = processor
        self.max_pixels, self.min_pixels = max_pixels, min_pixels

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        img = Image.open(rec["image"]).convert("RGB")
        answer = target_text(rec, self.max_pixels, self.min_pixels)

        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt_text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + answer + "<|im_end|>"

        full = self.proc(text=[full_text], images=[img], return_tensors="pt")
        # prompt 부분 길이를 같은 이미지로 재계산해야 <|image_pad|> 확장 길이가 일치한다
        pl = self.proc(text=[prompt_text], images=[img], return_tensors="pt")["input_ids"].shape[1]

        ids = full["input_ids"][0]
        labels = ids.clone()
        labels[:pl] = -100  # 프롬프트에는 loss를 걸지 않는다
        return {
            "input_ids": ids,
            "attention_mask": full["attention_mask"][0],
            "labels": labels,
            "pixel_values": full["pixel_values"],
            "image_grid_thw": full["image_grid_thw"],
        }


def collate(batch, pad_id):
    L = max(b["input_ids"].shape[0] for b in batch)
    ids, att, lab = [], [], []
    for b in batch:
        n = L - b["input_ids"].shape[0]
        ids.append(torch.cat([b["input_ids"], torch.full((n,), pad_id, dtype=torch.long)]))
        att.append(torch.cat([b["attention_mask"], torch.zeros(n, dtype=torch.long)]))
        lab.append(torch.cat([b["labels"], torch.full((n,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(att),
        "labels": torch.stack(lab),
        # Qwen은 패치를 평탄화해 담으므로 dim0 concat이 맞다
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], dim=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="vlm_prepare.py 산출 디렉토리(train.jsonl 포함)")
    ap.add_argument("--out", default="runs/vlm/qwen25vl3b")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=1280 * 736,
                    help="YOLO의 1280x736 rect와 맞춘 기본값. 낮추면 소형 객체가 죽는다")
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--dump-sample", type=int, default=0,
                    help="학습 안 하고 샘플 K개의 최종 문자열/좌표만 출력(사전 검증)")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments

    proc = AutoProcessor.from_pretrained(
        args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )

    if args.dump_sample:
        ds = DetDataset(Path(args.data) / "train.jsonl", proc, args.max_pixels, args.min_pixels)
        for i in range(min(args.dump_sample, len(ds))):
            rec = ds.recs[i]
            nh, nw = smart_resize(rec["height"], rec["width"],
                                  min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            print(f"--- {Path(rec['image']).name}  {rec['width']}x{rec['height']} -> {nw}x{nh}")
            print("  GT(orig) :", [o["bbox"] for o in rec["objects"]])
            print("  TARGET   :", target_text(rec, args.max_pixels, args.min_pixels))
            print(f"  tokens   : {ds[i]['input_ids'].shape[0]}  (vision {ds[i]['image_grid_thw'].prod() // 4})")
        print("\n>> nw/nh가 1280x736 근처인지, TARGET 좌표가 그 범위 안인지 확인할 것.")
        return

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False

    # ViT는 동결. 소형 객체 때문에 나중에 풀고 싶어지겠지만, 먼저 baseline부터.
    for p in model.visual.parameters():
        p.requires_grad = False

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    train_ds = DetDataset(Path(args.data) / "train.jsonl", proc,
                          args.max_pixels, args.min_pixels, shuffle_seed=0)
    print(f"train samples: {len(train_ds)}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=3,     # §13: 마지막 하나만 남기는 사고 방지
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )
    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    Trainer(model=model, args=targs, train_dataset=train_ds,
            data_collator=lambda b: collate(b, pad_id)).train()

    model.save_pretrained(Path(args.out) / "adapter_final")
    proc.save_pretrained(Path(args.out) / "adapter_final")
    print("saved:", Path(args.out) / "adapter_final")


if __name__ == "__main__":
    main()
