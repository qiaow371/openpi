"""冻视觉、只训语言头：Task + 图 → 短 subtask。验证用生成匹配。

从 annotate_pipeline 拷到 OpenPI，不改 JAX π0.5 动作训练。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROMPT = "Task: {task}\nWhat is the current subtask?\nSubtask:"
_WS = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.lower().strip()).strip()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


class SubtaskSheetDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path):
        self.rows = [r for r in rows if r.get("sheet")]
        self.root = root
        if not self.rows:
            raise SystemExit("jsonl 里没有 sheet。先在 annotate_pipeline prepare --sheets 再 compose")

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


def _is_vision_param(name: str) -> bool:
    n = name.lower()
    keys = ("vision_tower", "vision_model", "visual", "siglip", "vision_encoder", "img.")
    return any(k in n for k in keys)


def freeze_vision(model) -> tuple[int, int]:
    n_freeze = n_train = 0
    for name, param in model.named_parameters():
        if _is_vision_param(name):
            param.requires_grad = False
            n_freeze += param.numel()
        else:
            n_train += param.numel()
    return n_freeze, n_train


def load_vlm(model_path: str, device: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    return processor, model


def _collate_paligemma(processor, batch: list[dict], device: str):
    packed = processor(
        images=[b["image"] for b in batch],
        text=[b["prompt"] for b in batch],
        suffix=[b["subtask"] for b in batch],
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in packed.items()}


def _collate_chat(processor, batch: list[dict], device: str):
    texts, images = [], []
    for b in batch:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": b["image"]},
                    {"type": "text", "text": b["prompt"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": b["subtask"]}]},
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        images.append(b["image"])
    packed = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = packed["input_ids"].clone()
    labels[packed["attention_mask"] == 0] = -100
    packed["labels"] = labels
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in packed.items()}


def make_collate(processor, device: str):
    use_suffix = hasattr(processor, "suffix") or processor.__class__.__name__.lower().startswith("pali")

    def _fn(batch: list[dict]):
        if use_suffix:
            try:
                return _collate_paligemma(processor, batch, device)
            except TypeError:
                pass
        return _collate_chat(processor, batch, device)

    return _fn


@torch.no_grad()
def generate_one(processor, model, image: Image.Image, prompt: str, max_new_tokens: int = 32) -> str:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        packed = processor(images=image, text=prompt, return_tensors="pt")
        packed = {k: v.to(device) if hasattr(v, "to") else v for k, v in packed.items()}
        out = model.generate(**packed, max_new_tokens=max_new_tokens, do_sample=False)
        in_len = packed["input_ids"].shape[1]
        text = processor.batch_decode(out[:, in_len:], skip_special_tokens=True)[0]
    except Exception:  # noqa: BLE001
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        packed = processor(text=[text_in], images=[image], return_tensors="pt")
        packed = {k: v.to(device) if hasattr(v, "to") else v for k, v in packed.items()}
        out = model.generate(**packed, max_new_tokens=max_new_tokens, do_sample=False)
        in_len = packed["input_ids"].shape[1]
        text = processor.batch_decode(out[:, in_len:], skip_special_tokens=True)[0]
    if was_training:
        model.train()
    return text.strip().split("\n")[0].strip()


def closest_label(pred: str, templates: dict[str, str]) -> str | None:
    p = normalize(pred)
    if not p:
        return None
    best, best_d = None, 10**9
    for label, tmpl in templates.items():
        t = normalize(tmpl.replace("{hand}", "arm"))
        if t in p or p in t:
            return label
        d = abs(len(p) - len(t)) + sum(a != b for a, b in zip(p, t, strict=False))
        if d < best_d:
            best, best_d = label, d
    return best


def evaluate(processor, model, rows: list[dict], root: Path, templates: dict[str, str]) -> dict:
    n = exact = label_hit = 0
    by_label: dict[str, Counter] = defaultdict(Counter)
    preds = []
    for row in rows:
        if not row.get("sheet"):
            continue
        image = Image.open(root / row["sheet"]).convert("RGB")
        pred = generate_one(processor, model, image, PROMPT.format(task=row["task"]))
        gold = row["subtask"]
        n += 1
        ex = normalize(pred) == normalize(gold)
        mapped = closest_label(pred, templates)
        lab = mapped == row["label"]
        exact += int(ex)
        label_hit += int(lab)
        by_label[row["label"]]["n"] += 1
        by_label[row["label"]]["exact"] += int(ex)
        by_label[row["label"]]["label"] += int(lab)
        preds.append(
            {
                "episode": row["episode"],
                "label": row["label"],
                "gold": gold,
                "pred": pred,
                "exact": ex,
                "label_match": lab,
            }
        )
    return {
        "n": n,
        "exact_match": (exact / n) if n else 0.0,
        "label_acc": (label_hit / n) if n else 0.0,
        "by_label": {
            k: {
                "n": v["n"],
                "exact_match": v["exact"] / v["n"],
                "label_acc": v["label"] / v["n"],
            }
            for k, v in by_label.items()
        },
        "preds": preds,
    }


def save_vlm_curves(
    ckpt_dir: Path,
    *,
    losses: list[dict],
    history: list[dict],
    title: str = "vlm_subtask",
) -> Path | None:
    """Write train-loss + val exact/label_acc curves next to the checkpoint."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[vlm] skip curves: {exc}", flush=True)
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if losses:
        xs = [p["step"] for p in losses]
        ys = [p["loss"] for p in losses]
        axes[0].plot(xs, ys, color="C0", label="train loss")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].set_title("train")
        axes[0].legend()
    else:
        axes[0].set_title("train (no points)")
    if history:
        xs = [p["epoch"] for p in history]
        axes[1].plot(xs, [p["exact_match"] for p in history], marker="o", label="val exact_match")
        axes[1].plot(xs, [p["label_acc"] for p in history], marker="s", label="val label_acc")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("validation")
        axes[1].legend()
    else:
        axes[1].set_title("validation (no points)")
    fig.suptitle(title)
    fig.tight_layout()
    path = ckpt_dir / "curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_train(
    *,
    data_dir: Path,
    model_path: str,
    templates: dict[str, str],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    log_every: int,
    ckpt_dir: Path,
    split_dir: Path | None = None,
) -> None:
    split_dir = split_dir or data_dir
    train_rows = read_jsonl(split_dir / "vlm_train.jsonl")
    val_rows = read_jsonl(split_dir / "vlm_val.jsonl")
    if not train_rows:
        raise SystemExit(f"没有 {split_dir}/vlm_train.jsonl，先跑 compose")

    processor, model = load_vlm(model_path, device)
    model.train()
    n_freeze, n_train = freeze_vision(model)
    print(f"[vlm] freeze vision {n_freeze / 1e6:.1f}M, train {n_train / 1e6:.1f}M", flush=True)

    loader = DataLoader(
        SubtaskSheetDataset(train_rows, data_dir),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate(processor, device),
        num_workers=0,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    step = 0
    history = []
    losses: list[dict] = []

    for epoch in range(epochs):
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            step += 1
            loss_f = float(loss)
            losses.append({"epoch": epoch, "step": step, "loss": loss_f})
            if step % log_every == 0:
                print(f"[vlm] epoch={epoch} step={step} loss={loss_f:.4f}", flush=True)

        metrics = evaluate(processor, model, val_rows or train_rows[:8], data_dir, templates)
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
        print(f"[vlm] val epoch={epoch} exact={score:.3f} label_acc={metrics['label_acc']:.3f}", flush=True)
        write_json(ckpt_dir / "vlm_val_latest.json", {k: v for k, v in metrics.items() if k != "preds"})
        if score >= best:
            best = score
            processor.save_pretrained(ckpt_dir / "best")
            model.save_pretrained(ckpt_dir / "best")
            write_json(ckpt_dir / "best_metrics.json", {"epoch": epoch, "exact_match": best})
        write_json(
            ckpt_dir / "train_history.json",
            {
                "best_exact_match": best,
                "split_dir": str(split_dir),
                "n_train": len(train_rows),
                "n_val": len(val_rows),
                "losses": losses,
                "history": history,
            },
        )
        save_vlm_curves(ckpt_dir, losses=losses, history=history, title=f"vlm_subtask {data_dir.name}")

    processor.save_pretrained(ckpt_dir / "last")
    model.save_pretrained(ckpt_dir / "last")
    plot = save_vlm_curves(ckpt_dir, losses=losses, history=history, title=f"vlm_subtask {data_dir.name}")
    print(f"[vlm] saved last + best (exact={best:.3f}) -> {ckpt_dir}" + (f" curves={plot}" if plot else ""))


def run_val(
    *,
    data_dir: Path,
    model_path: str,
    templates: dict[str, str],
    device: str,
    out_path: Path,
    split_dir: Path | None = None,
) -> dict:
    split_dir = split_dir or data_dir
    rows = read_jsonl(split_dir / "vlm_val.jsonl")
    if not rows:
        raise SystemExit(f"没有 {split_dir}/vlm_val.jsonl")
    processor, model = load_vlm(model_path, device)
    model.eval()
    metrics = evaluate(processor, model, rows, data_dir, templates)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with (out_path.parent / "vlm_val_preds.jsonl").open("w") as f:
        for row in metrics["preds"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {k: v for k, v in metrics.items() if k != "preds"}
    write_json(out_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
