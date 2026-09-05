# 标注 → OpenPI π0.5 训练

`annotate_pipeline` 的切段/模板 **不是** 训练 `repo_id`。要写进 LeRobot 数据集的 AgiBot sidecar，然后照常 `train_from_yaml.py`。

```text
annotate prepare  →  outputs/<id>/segments.jsonl
compose           →  outputs/<id>/cot.jsonl          # FSM 模板短句 ˆℓ
export            →  <lerobot>/meta/episodes_detailed_task.jsonl
train_from_yaml   →  有 span 的帧 prompt=action_text，其余用 tasks.jsonl
```

`vlm_subtask train` 是可选的 HF PaliGemma 旁路（冻视觉、训语言头），**不是** JAX 动作训练。

## 接到 JAX 训练（这条才是 π0.5）

```bash
# 1) 切段（全量；不要 --limit-ep）
bash /home/xk/VLA/annotate_pipeline/deploy.sh prepare --dataset 00086

# 2) 模板短句 + 写进数据集 sidecar
bash /home/xk/VLA/annotate_pipeline/deploy.sh export --dataset 00086

# 3) 改 configs/train_pi05.yaml 的 data.repo_id 指向该 LeRobot 根目录
#    prompt_from_subtask 已打开。确认：
cd /home/xk/VLA/openpi/cigai-agilex-train-openpi
PYTHONPATH=src:. .venv/bin/python scripts/train_from_yaml.py \
  --config configs/train_pi05.yaml --dry-run
# 看：prompt_from_subtask: true
#     subtask_sidecar: .../episodes_detailed_task.jsonl (N/M episodes, ...)

# 4) 开训
.venv/bin/python scripts/train_from_yaml.py --config configs/train_pi05.yaml
```

启动后日志应有：`Loading AgiBot-style subtask prompts from ... (N/M episodes, ... spans)`。

覆盖率不足时（例如 2/300）绝大多数帧仍是全局 task，对照实验无意义，先把 prepare 跑全。

sidecar（不改 parquet）：

```text
meta/task_info/task_00086.json
meta/episodes_detailed_task.jsonl    # dataloader 主读取
meta/subtask_spans.jsonl
```

## 可选：HF VLM 旁路

```bash
cd /home/xk/VLA/openpi/cigai-agilex-train-openpi
PYTHONPATH=. .venv/bin/python -m vlm_subtask train \
  --config vlm_subtask/configs/00086.yaml \
  --data-dir /home/xk/VLA/annotate_pipeline/outputs/00086 \
  --model google/paligemma-3b-pt-224
```
