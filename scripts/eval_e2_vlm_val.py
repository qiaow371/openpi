#!/usr/bin/env python3
"""e2 PaliGemma → 冻结 val 155 段：生成短 subtask，写 exact_match / label_acc。

从 π0.5 权重里抽出 paligemma（PaliGemmaForConditionalGeneration.generate）。
有 e2 step ckpt 用 ckpt，否则用 pi05_fixed（e2 起点）。
不占训练 GPU：默认 CPU。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_paths() -> Path:
    repo = _repo()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    return repo


def latest_e2_weights(ckpt_root: Path) -> tuple[Path | None, int | None]:
    from openpi.training import train_log as _train_log

    if not ckpt_root.is_dir():
        return None, None
    steps = _train_log.list_checkpoint_steps(ckpt_root)
    if not steps:
        return None, None
    latest = max(steps)
    step_dir = _train_log.resolve_step_dir(ckpt_root, latest, 19700)
    weights = _train_log.get_pretrained_model_dir(step_dir)
    if (weights / "model.safetensors").is_file():
        return weights, int(latest)
    return None, None


def build_paligemma():
    from transformers.models.auto import CONFIG_MAPPING
    from transformers import PaliGemmaForConditionalGeneration

    import openpi.models.gemma as _gemma

    vlm = _gemma.get_config("gemma_2b")
    cfg = CONFIG_MAPPING["paligemma"]()
    cfg._vocab_size = 257152  # noqa: SLF001
    cfg.image_token_index = 257152
    cfg.text_config.hidden_size = vlm.width
    cfg.text_config.intermediate_size = vlm.mlp_dim
    cfg.text_config.num_attention_heads = vlm.num_heads
    cfg.text_config.head_dim = vlm.head_dim
    cfg.text_config.num_hidden_layers = vlm.depth
    cfg.text_config.num_key_value_heads = vlm.num_kv_heads
    cfg.text_config.hidden_activation = "gelu_pytorch_tanh"
    cfg.text_config.torch_dtype = "float32"
    cfg.text_config.vocab_size = 257152
    cfg.text_config.use_adarms = False
    cfg.text_config.adarms_cond_dim = None
    cfg.vision_config.intermediate_size = 4304
    cfg.vision_config.projection_dim = 2048
    cfg.vision_config.projector_hidden_act = "gelu_fast"
    cfg.vision_config.torch_dtype = "float32"
    return PaliGemmaForConditionalGeneration(config=cfg)


def load_paligemma(weight_dir: Path, device: torch.device):
    import safetensors.torch

    model = build_paligemma()
    src = safetensors.torch.load_file(str(weight_dir / "model.safetensors"))
    prefix = "paligemma_with_expert.paligemma."
    sub = {}
    for key, tensor in src.items():
        if key.startswith(prefix):
            sub[key[len(prefix) :]] = tensor
    if not sub:
        raise SystemExit(f"{weight_dir} 没有 paligemma_with_expert.paligemma.*")
    dest = model.state_dict()
    remapped = {}
    for key, tensor in sub.items():
        target = key
        if target not in dest:
            alt = target.replace(".model.language_model.", ".language_model.")
            if alt in dest:
                target = alt
        if target not in dest and target.startswith("language_model."):
            alt = "model." + target
            if alt in dest:
                target = alt
        if target not in dest and target.startswith("model.language_model."):
            alt = target.replace("model.language_model.", "language_model.", 1)
            if alt in dest:
                target = alt
        if target in dest and tuple(dest[target].shape) == tuple(tensor.shape):
            remapped[target] = tensor
    # tied embed: HF keeps model.language_model.embed_tokens; pi05 often only has language_model.embed_tokens
    if (
        "model.language_model.embed_tokens.weight" in dest
        and "model.language_model.embed_tokens.weight" not in remapped
        and "language_model.embed_tokens.weight" in remapped
    ):
        remapped["model.language_model.embed_tokens.weight"] = remapped["language_model.embed_tokens.weight"]
    if (
        "language_model.embed_tokens.weight" in dest
        and "language_model.embed_tokens.weight" not in remapped
        and "model.language_model.embed_tokens.weight" in remapped
    ):
        remapped["language_model.embed_tokens.weight"] = remapped["model.language_model.embed_tokens.weight"]
    # π0.5 部分 ckpt 只存 lm_head（与 embed_tokens tied）
    if (
        "model.language_model.embed_tokens.weight" in dest
        and "model.language_model.embed_tokens.weight" not in remapped
        and "lm_head.weight" in remapped
        and tuple(dest["model.language_model.embed_tokens.weight"].shape)
        == tuple(remapped["lm_head.weight"].shape)
    ):
        remapped["model.language_model.embed_tokens.weight"] = remapped["lm_head.weight"]
        logging.info("filled embed_tokens from tied lm_head")
    if (
        "language_model.embed_tokens.weight" in dest
        and "language_model.embed_tokens.weight" not in remapped
        and "lm_head.weight" in remapped
        and tuple(dest["language_model.embed_tokens.weight"].shape)
        == tuple(remapped["lm_head.weight"].shape)
    ):
        remapped["language_model.embed_tokens.weight"] = remapped["lm_head.weight"]
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    logging.info("paligemma load missing=%s unexpected=%s used=%s", len(missing), len(unexpected), len(remapped))
    if missing[:8]:
        logging.warning("missing first: %s", list(missing)[:8])
    model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.eval()
    return model


IMAGE_TOKEN_ID = 257152
N_IMAGE_TOKENS = 256


def load_sentencepiece(tokenizer_model: Path):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_model))
    return sp


def preprocess_image(path: Path, device: torch.device) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)


@torch.no_grad()
def generate_subtask(model, sp, image_path: Path, prompt: str, device: torch.device, max_new_tokens: int = 32) -> str:
    pixels = preprocess_image(image_path, device)
    text_ids = list(sp.encode(prompt.strip(), add_bos=True)) + list(sp.encode("\n"))
    input_ids = torch.tensor([[IMAGE_TOKEN_ID] * N_IMAGE_TOKENS + text_ids], device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids)
    out = model.generate(
        pixel_values=pixels,
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_ids = [int(i) for i in out[0, input_ids.shape[1] :].tolist() if 0 <= int(i) < sp.GetPieceSize()]
    text = sp.decode(new_ids).strip().split("\n")[0].strip()
    return text


def evaluate_sp(model, sp, rows: list[dict], root: Path, templates: dict[str, str], device: torch.device) -> dict:
    from collections import Counter, defaultdict

    from vlm_subtask.train_val import PROMPT, closest_label, normalize

    n = exact = label_hit = 0
    by_label: dict[str, Counter] = defaultdict(Counter)
    preds = []
    for row in rows:
        sheet = row.get("sheet")
        if not sheet:
            continue
        pred = generate_subtask(model, sp, root / sheet, PROMPT.format(task=row["task"]), device)
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
        if n % 10 == 0:
            logging.info("val %s/%s last pred=%r gold=%r", n, len(rows), pred, gold)
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


def write_record(summary: dict) -> None:
    rec = Path("/nvme1n1/vlm-sft/outputs/annotate/pipeline_record.json")
    hist = Path("/nvme1n1/vlm-sft/outputs/annotate/pipeline_record_history.jsonl")
    mounted = Path("/workspace/openpi/outputs/annotate/pipeline_record.json")
    local = _repo().parents[1] / "annotate_pipeline" / "outputs" / "pipeline_record.json"
    local_hist = local.parent / "pipeline_record_history.jsonl"
    targets = [p for p in (rec, mounted, local) if p.is_file()]
    if not targets:
        logging.warning("pipeline_record.json not found")
        return
    for path in targets:
        obj = json.loads(path.read_text())
        obj["updated_at"] = datetime.now().isoformat(timespec="seconds")
        e2 = obj.setdefault("experiments", {}).setdefault("e2_subtask_ce", {})
        e2["can_generate_subtask"] = True
        e2["val_after_train"] = {
            "subtask_text_em": True,
            "how": "CE-trained paligemma generate on splits_frozen val 155",
            "split": "splits_frozen 00036 val 155 / 23 ep",
            "note": "subtask 是 CE 标签；不是 FM e2_pg_only",
        }
        obj["e2_vlm_val"] = summary
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
        if path == rec:
            h = hist
        elif path == mounted:
            h = mounted.parent / "pipeline_record_history.jsonl"
        else:
            h = local_hist
        with h.open("a") as f:
            f.write(json.dumps({"archived_at": obj["updated_at"], "kind": "e2_vlm_val", "record": summary}, ensure_ascii=False) + "\n")
        logging.info("updated %s", path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weight-dir", default="", help="含 model.safetensors 的目录；空则优先 e2_subtask_ce/best")
    p.add_argument(
        "--e2-ckpt",
        default="/workspace/openpi/cigai_train/openpi-jiangsuan/checkpoints/e2_subtask_ce_00036/best",
        help="CE best 目录，或旧 FM e2 的 checkpoints 根（已废弃）",
    )
    p.add_argument("--pi05-fixed", default="/workspace/openpi/models/pi05_fixed")
    p.add_argument("--data-dir", default="/workspace/openpi/outputs/annotate/00036")
    p.add_argument("--split-dir", default="/workspace/openpi/outputs/annotate/00036/splits_frozen")
    p.add_argument("--out", default="/workspace/openpi/outputs/annotate/00036/e2_vlm_val")
    p.add_argument("--config", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _add_paths()
    from vlm_subtask.train_val import read_jsonl, write_json

    repo = _repo()
    cfg_path = Path(args.config) if args.config else repo / "vlm_subtask/configs/00036.yaml"
    import yaml

    templates = (yaml.safe_load(cfg_path.read_text()) or {}).get("subtask_templates") or {}
    weight_dir = Path(args.weight_dir) if args.weight_dir else None
    step = None
    source = "explicit"
    if weight_dir is None:
        ce_best = Path(args.e2_ckpt)
        if (ce_best / "model.safetensors").is_file():
            weight_dir = ce_best
            source = "e2_subtask_ce_best"
            step = "ce_best"
        else:
            e2_dir, step = latest_e2_weights(Path(args.e2_ckpt))
            if e2_dir is not None:
                weight_dir = e2_dir
                source = "e2_ckpt"
            else:
                weight_dir = Path(args.pi05_fixed)
                source = "pi05_fixed_e2_init"
                step = 0
    if not (weight_dir / "model.safetensors").is_file():
        raise SystemExit(f"没有 model.safetensors: {weight_dir}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    logging.info("weights=%s source=%s step=%s device=%s", weight_dir, source, step, device)
    model = load_paligemma(weight_dir, device)
    tok_path = Path(args.pi05_fixed) / "tokenizer.model"
    if not tok_path.is_file():
        tok_path = weight_dir / "tokenizer.model"
    sp = load_sentencepiece(tok_path)

    data_dir = Path(args.data_dir)
    split_dir = Path(args.split_dir)
    rows = read_jsonl(split_dir / "vlm_val.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"没有 {split_dir}/vlm_val.jsonl")

    metrics = evaluate_sp(model, sp, rows, data_dir, templates, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "vlm_val_preds.jsonl").open("w") as f:
        for row in metrics["preds"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {k: v for k, v in metrics.items() if k != "preds"}
    summary.update(
        {
            "kind": "e2_vlm_subtask_val",
            "host": "jiangsuan-16",
            "weight_dir": str(weight_dir),
            "source": source,
            "step": step,
            "n_rows": len(rows),
            "device": str(device),
            "split_dir": str(split_dir),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    write_json(out / "vlm_val_summary.json", summary)
    write_record(summary)
    logging.info(
        "e2 vlm val n=%s exact=%.3f label_acc=%.3f source=%s step=%s -> %s",
        summary.get("n"),
        summary.get("exact_match") or 0.0,
        summary.get("label_acc") or 0.0,
        source,
        step,
        out / "vlm_val_summary.json",
    )


if __name__ == "__main__":
    main()
