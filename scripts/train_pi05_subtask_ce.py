#!/usr/bin/env python3
"""π0.5 PaliGemma：冻视觉，subtask 短句 CE（不是动作 FM）。

从 pi05 权重抽出 paligemma，用 sheets + vlm_train/val.jsonl：
  Task + 图 → Subtask  （lm_head CE）
每个 epoch 做 generate exact_match / label_acc。

用法（江算 openpi_train）:
  python3 scripts/train_pi05_subtask_ce.py \\
    --weight-dir /workspace/openpi/models/pi05_fixed \\
    --data-dir /workspace/openpi/outputs/annotate/00036 \\
    --split-dir /workspace/openpi/outputs/annotate/00036/splits_frozen \\
    --ckpt-dir /workspace/openpi/cigai_train/openpi-jiangsuan/checkpoints/e2_subtask_ce_00036 \\
    --device cuda:0 --epochs 10 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROMPT = "Task: {task}\nWhat is the current subtask?\nSubtask:"
IMAGE_TOKEN_ID = 257152
N_IMAGE_TOKENS = 256


def _add_paths() -> Path:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    return repo


class SheetDS(Dataset):
    def __init__(self, rows: list[dict], root: Path):
        self.rows = [r for r in rows if r.get("sheet")]
        self.root = root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = Image.open(self.root / row["sheet"]).convert("RGB")
        return {
            "image": image,
            "prompt": PROMPT.format(task=row["task"]),
            "subtask": row["subtask"],
            "label": row["label"],
            "episode": row["episode"],
        }


def freeze_vision(model) -> tuple[int, int]:
    n_f = n_t = 0
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in ("vision_tower", "vision_model", "multi_modal_projector")):
            p.requires_grad = False
            n_f += p.numel()
        else:
            n_t += p.numel()
    return n_f, n_t


def preprocess_image(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    import numpy as np

    img = image.resize((224, 224), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=dtype)


def encode_batch(sp, images: list[Image.Image], prompts: list[str], subtasks: list[str], device, dtype):
    """prefix = <image>*256 + prompt+\n ; labels only on subtask tokens (+ eos)."""
    pixel_list = [preprocess_image(im, device, dtype).squeeze(0) for im in images]
    pixels = torch.stack(pixel_list, dim=0)

    input_rows = []
    label_rows = []
    for prompt, sub in zip(prompts, subtasks):
        prefix = list(sp.encode(prompt.strip(), add_bos=True)) + list(sp.encode("\n"))
        suffix = list(sp.encode(sub.strip(), add_bos=False)) + [sp.eos_id()]
        ids = [IMAGE_TOKEN_ID] * N_IMAGE_TOKENS + prefix + suffix
        labs = [-100] * (N_IMAGE_TOKENS + len(prefix)) + suffix
        input_rows.append(ids)
        label_rows.append(labs)

    max_len = max(len(r) for r in input_rows)
    pad_id = 0
    input_ids = torch.full((len(input_rows), max_len), pad_id, dtype=torch.long, device=device)
    labels = torch.full((len(input_rows), max_len), -100, dtype=torch.long, device=device)
    attn = torch.zeros((len(input_rows), max_len), dtype=torch.long, device=device)
    for i, (ids, labs) in enumerate(zip(input_rows, label_rows)):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        labels[i, : len(labs)] = torch.tensor(labs, dtype=torch.long, device=device)
        attn[i, : len(ids)] = 1
    return {"pixel_values": pixels, "input_ids": input_ids, "attention_mask": attn, "labels": labels}


@torch.no_grad()
def generate_one(model, sp, image: Image.Image, prompt: str, device, dtype, max_new=32) -> str:
    pixels = preprocess_image(image, device, dtype)
    text_ids = list(sp.encode(prompt.strip(), add_bos=True)) + list(sp.encode("\n"))
    input_ids = torch.tensor([[IMAGE_TOKEN_ID] * N_IMAGE_TOKENS + text_ids], device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids)
    out = model.generate(
        pixel_values=pixels,
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new,
        do_sample=False,
    )
    new_ids = [int(i) for i in out[0, input_ids.shape[1] :].tolist() if 0 <= int(i) < sp.GetPieceSize()]
    return sp.decode(new_ids).strip().split("\n")[0].strip()


def evaluate(model, sp, rows, root, templates, device, dtype, limit=0) -> dict:
    from vlm_subtask.train_val import closest_label, normalize

    model.eval()
    n = exact = label_hit = 0
    preds = []
    for row in rows:
        if not row.get("sheet"):
            continue
        if limit and n >= limit:
            break
        image = Image.open(root / row["sheet"]).convert("RGB")
        pred = generate_one(model, sp, image, PROMPT.format(task=row["task"]), device, dtype)
        gold = row["subtask"]
        n += 1
        ex = normalize(pred) == normalize(gold)
        mapped = closest_label(pred, templates)
        lab = mapped == row["label"]
        exact += int(ex)
        label_hit += int(lab)
        preds.append({"episode": row["episode"], "label": row["label"], "gold": gold, "pred": pred, "exact": ex, "label_match": lab})
        if n % 20 == 0:
            logging.info("val %s pred=%r gold=%r", n, pred, gold)
    return {
        "n": n,
        "exact_match": (exact / n) if n else 0.0,
        "label_acc": (label_hit / n) if n else 0.0,
        "preds": preds,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weight-dir", required=True)
    p.add_argument("--tokenizer", default="")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--config", default="")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-limit", type=int, default=0, help="0=full val")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    repo = _add_paths()
    from scripts.eval_e2_vlm_val import load_paligemma, load_sentencepiece
    from vlm_subtask.train_val import read_jsonl, write_json, save_vlm_curves

    import yaml

    cfg_path = Path(args.config) if args.config else repo / "vlm_subtask/configs/00036.yaml"
    templates = (yaml.safe_load(cfg_path.read_text()) or {}).get("subtask_templates") or {}

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    weight_dir = Path(args.weight_dir)
    tok = Path(args.tokenizer) if args.tokenizer else weight_dir / "tokenizer.model"
    if not tok.is_file():
        tok = Path("/workspace/openpi/models/pi05_fixed/tokenizer.model")

    model = load_paligemma(weight_dir, device)
    model.train()
    n_f, n_t = freeze_vision(model)
    logging.info("freeze vision %.1fM, train language %.1fM", n_f / 1e6, n_t / 1e6)
    sp = load_sentencepiece(tok)

    split_dir = Path(args.split_dir)
    data_dir = Path(args.data_dir)
    train_rows = read_jsonl(split_dir / "vlm_train.jsonl")
    val_rows = read_jsonl(split_dir / "vlm_val.jsonl")
    train_ds = SheetDS(train_rows, data_dir)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: b, num_workers=0)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    step = 0
    losses = []
    history = []

    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            packed = encode_batch(
                sp,
                [b["image"] for b in batch],
                [b["prompt"] for b in batch],
                [b["subtask"] for b in batch],
                device,
                dtype,
            )
            opt.zero_grad(set_to_none=True)
            out = model(
                pixel_values=packed["pixel_values"],
                input_ids=packed["input_ids"],
                attention_mask=packed["attention_mask"],
                labels=packed["labels"],
            )
            loss = out.loss
            loss.backward()
            opt.step()
            step += 1
            loss_f = float(loss.detach())
            losses.append({"epoch": epoch, "step": step, "loss": loss_f})
            if step % args.log_every == 0:
                logging.info("epoch=%s step=%s loss=%.4f", epoch, step, loss_f)

        metrics = evaluate(model, sp, val_rows, data_dir, templates, device, dtype, limit=args.val_limit)
        score = float(metrics["exact_match"])
        history.append(
            {
                "epoch": epoch,
                "step": step,
                "exact_match": score,
                "label_acc": metrics["label_acc"],
                "n": metrics["n"],
            }
        )
        logging.info(
            "val epoch=%s exact=%.3f label_acc=%.3f n=%s",
            epoch,
            score,
            metrics["label_acc"],
            metrics["n"],
        )
        write_json(ckpt_dir / "vlm_val_latest.json", {k: v for k, v in metrics.items() if k != "preds"})
        if score >= best:
            best = score
            out_dir = ckpt_dir / "best"
            out_dir.mkdir(parents=True, exist_ok=True)
            # save with pi05 prefix so splice back is easy
            state = {f"paligemma_with_expert.paligemma.{k}": v.detach().cpu() for k, v in model.state_dict().items()}
            import safetensors.torch

            safetensors.torch.save_file(state, str(out_dir / "model.safetensors"))
            write_json(out_dir / "best_metrics.json", {"epoch": epoch, "exact_match": best, "label_acc": metrics["label_acc"]})
        write_json(
            ckpt_dir / "train_history.json",
            {
                "kind": "pi05_subtask_ce",
                "best_exact_match": best,
                "weight_dir": str(weight_dir),
                "n_train": len(train_rows),
                "n_val": len(val_rows),
                "freeze_vision": True,
                "loss": "language_ce",
                "not_flow_matching": True,
                "losses": losses,
                "history": history,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        save_vlm_curves(ckpt_dir, losses=losses, history=history, title="pi05_subtask_ce 00036")

    # last
    last = ckpt_dir / "last"
    last.mkdir(parents=True, exist_ok=True)
    import safetensors.torch

    state = {f"paligemma_with_expert.paligemma.{k}": v.detach().cpu() for k, v in model.state_dict().items()}
    safetensors.torch.save_file(state, str(last / "model.safetensors"))
    logging.info("done best_exact=%.3f -> %s", best, ckpt_dir)


if __name__ == "__main__":
    main()
