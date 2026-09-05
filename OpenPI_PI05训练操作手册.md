# OpenPI π₀.₅ 训练操作手册

前提：训练机已装好 NVIDIA 驱动 / CUDA，SSH 能登录。下面整段复制粘贴即可（路径按你的机器改）。

分清两样东西：

| 名称 | 是什么 | 怎么配 |
|------|--------|--------|
| LeRobot 依赖 | 随仓库用 `uv` 安装 | 第 3～4 步 |
| `data.repo_id` | 你的 **训练数据集** 路径 | 在 `configs/train_pi05.yaml` 里改 |

默认用 YAML + `train_from_yaml.py`（推荐）。`framework: jax` 为默认；改成 `pytorch` 即可走 PyTorch。

---

## 1. 拉代码

```bash
cd /home/agilex/cigai_train/openpi
git checkout openpi
git pull origin openpi
```

没有仓库时：

```bash
mkdir -p /home/agilex/cigai_train
cd /home/agilex/cigai_train
git clone git@gitlab.cigai.cn:center_embodied_ai/dept_application_delivery/cigai-agilex-train.git openpi
cd openpi
git checkout openpi
git submodule update --init --recursive
```

---

## 2. 改数据集路径（必须）

```bash
cd /home/agilex/cigai_train/openpi
nano configs/train_pi05.yaml
```

改成你的数据绝对路径（目录里要有 `meta/info.json`），并取消注释：

```yaml
data:
  repo_id: /home/agilex/dataset-pika_aloha-joint
  assets:
    assets_dir: /home/agilex/dataset-pika_aloha-joint
    asset_id: "."
```

`assets_dir` 一般和 `repo_id` 写成一样。`asset_id: "."` 表示 `norm_stats.json` 就在该目录下。

检查：

```bash
ls /home/agilex/dataset-pika_aloha-joint/meta/info.json
```

16D 双臂+夹爪可改用 `configs/train_pi05_tongbot.yaml`（同样要填 `repo_id` / `assets`）。

---

## 3. 环境配置（新环境做一次；进目录后用 uv）

已用 [uv](https://docs.astral.sh/uv/) 管理依赖。安装 uv 后：

```bash
cd /home/agilex/cigai_train/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install pyyaml
```

检查：

```bash
uv run python -c "import openpi, torch, jax; print('ok', torch.__version__)"
```

后面算 norm / 训练都用 `uv run ...`（不必另开 conda）。

---

## 4. 算归一化（每个新数据集做一次）

`--dataset-dir` 必须和 YAML 里 `repo_id` 相同。成功会在**数据集根目录**写出 `norm_stats.json`（不是 `./assets`）。

```bash
cd /home/agilex/cigai_train/openpi
uv run scripts/compute_norm_stats.py \
  --config-name pi05_aloha \
  --dataset-dir /home/xk/dataset/00031_20260729_RM_01_yj_v21
```

检查：

```bash
ls /home/agilex/dataset-pika_aloha-joint/norm_stats.json
```

训练时只**加载**这份文件，并拷进 checkpoint 的 `assets/` 给推理用；不会在训练时现算。

---

## 5. 选卡

```bash
nvidia-smi
```

单卡（改数字换卡）：

```bash
export CUDA_VISIBLE_DEVICES=0
```

多卡：

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

JAX 训练建议同时设显存占用：

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

多卡时 YAML 里 `batch_size` 要能被卡数整除；可用 `fsdp_devices` 做模型并行减单卡显存。

---

## 6. 开始训练

先 dry-run 看解析结果：

```bash
cd /home/agilex/cigai_train/openpi
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train_from_yaml.py --config configs/train_pi05.yaml --dry-run
```

正式开训：

```bash
cd /home/agilex/cigai_train/openpi
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train_from_yaml.py --config configs/train_pi05_tongbot.yaml
```

PyTorch 时把 YAML 里改成 `framework: pytorch`，命令相同。

成功标志：日志里出现 `Loaded norm stats`，然后有训练进度 / loss 日志。

模型目录示例：`checkpoints/pi05_chunk_*_lr_*_epochs_*[_bs_8...]_YYYYMMDD/`（省略 `exp_name` 时自动生成；数字超参相对默认有变才追加缩写）。

---

## 7. 下次继续训（最短）

```bash
cd /home/agilex/cigai_train/openpi
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train_from_yaml.py --config configs/train_pi05.yaml
```

断点续训：YAML 里设 `resume: true`，并把 `exp_name` 填成已有 run 名（不要用 `auto`）。

---

## 8. 出问题对照

| 现象 | 做法 |
|------|------|
| 找不到 `info.json` | 第 2 步改对 `repo_id` / `assets_dir` |
| `Normalization stats not found` / 跳过 norm | 第 4 步 `--dataset-dir` 与 `repo_id` 一致后再算；确认 `asset_id: "."` |
| `ModuleNotFoundError` / 依赖缺失 | 重做第 3 步 |
| JAX OOM | 减小 `batch_size`，或加大 `fsdp_devices`，或换更大显存卡 |
| PyTorch DataLoader 卡住 | YAML 里 `num_workers: 0` |
| dry-run 路径不对 | 确认改的是正在用的 YAML；`assets_base_dir: ./assets` 只是回退，正式流程靠 `data.assets` |

更细的说明见 [docs/norm_stats.md](docs/norm_stats.md)。
