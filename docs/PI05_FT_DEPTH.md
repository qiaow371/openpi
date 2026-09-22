# 本机 π0.5：RGB / DEPTH / FT 消融

分支 `train/pi05-rgb-depth-ft-yaml`（从 `train/pi05-ft-depth` 拉出）。**不改** 现用 RGB-only `pi05_aloha`（`front/left/right` 旧数据集）。

## 选项

三路 RGB 始终进网（`cam_mid` / `cam_left` / `cam_right`）。DEPTH 和 FT 是独立开关；DEPTH 三选一：HEAD / LEFT WRIST / RIGHT WRIST（一次只加一路 depth）。

| 选项 | DEPTH | FT | `config_name` | YAML |
|---|---|---|---|---|
| 1. 只训练 RGB | 无 | 关 | `pi05_aloha_rgb` | [`configs/train_pi05_rgb.yaml`](../configs/train_pi05_rgb.yaml) |
| 2a. RGB + DEPTH HEAD | `cam_mid` | 关 | `pi05_aloha_rgb_depth_head` | [`configs/train_pi05_rgb_depth_head.yaml`](../configs/train_pi05_rgb_depth_head.yaml) |
| 2b. RGB + DEPTH LEFT WRIST | `cam_left` | 关 | `pi05_aloha_rgb_depth_left` | [`configs/train_pi05_rgb_depth_left.yaml`](../configs/train_pi05_rgb_depth_left.yaml) |
| 2c. RGB + DEPTH RIGHT WRIST | `cam_right` | 关 | `pi05_aloha_rgb_depth_right` | [`configs/train_pi05_rgb_depth_right.yaml`](../configs/train_pi05_rgb_depth_right.yaml) |
| 3. RGB + FT | 无 | 开 | `pi05_aloha_rgb_ft` | [`configs/train_pi05_rgb_ft.yaml`](../configs/train_pi05_rgb_ft.yaml) |
| 4a. RGB + DEPTH HEAD + FT | `cam_mid` | 开 | `pi05_aloha_rgb_depth_head_ft` | [`configs/train_pi05_rgb_depth_head_ft.yaml`](../configs/train_pi05_rgb_depth_head_ft.yaml) |
| 4b. RGB + DEPTH LEFT + FT | `cam_left` | 开 | `pi05_aloha_rgb_depth_left_ft` | [`configs/train_pi05_rgb_depth_left_ft.yaml`](../configs/train_pi05_rgb_depth_left_ft.yaml) |
| 4c. RGB + DEPTH RIGHT + FT | `cam_right` | 开 | `pi05_aloha_rgb_depth_right_ft` | [`configs/train_pi05_rgb_depth_right_ft.yaml`](../configs/train_pi05_rgb_depth_right_ft.yaml) |

HEAD = `cam_mid`（头/胸口）。LEFT = `cam_left`。RIGHT = `cam_right`。不要混用 `pi05_aloha`（那套是 `front/left/right`）。

旧预设 `pi05_aloha_ft_depth` 仍在：三路 depth 全开 + FT。新实验请用上表，不要再用 [`configs/train_pi05_ft_depth.yaml`](../configs/train_pi05_ft_depth.yaml)。

## 数据管线

Depth **不走** 独立 `DepthEncoder`，也不在采集侧 `cv2.resize`。
`rs.align` 之后 depth 已与 RGB 同 HxW；训练时复制成 3 通道灰度，并入 `images`，和 RGB 一起进 **SigLIP → Gemma**。

SigLIP So400m/14 的输入仍是 **224×224**（`ResizeImages`，`resize_with_pad`）。RGB 本来就是这条；这不是把 320×240 depth 硬拉到 640×480。

FT 仍走 `Force6DEncoder`。RGB-only / RGB+DEPTH 预设把 `use_force6d_encoder=False`，Repack 也不带 `force6d`。

DEPTH 档会挂 `LoadSidecarDepthPNGs(depth_keys=...)`，只读选中的相机 PNG，不会把三路都扫进来。
现场盘文件名是 `frame_000000.png`，`meta.json` 模板是 `frame-000000.png`；loader **两种都认**。uint16 毫米按 0–4 m 拉到 8-bit 再进 SigLIP（不要 /65535，否则几乎全黑）。FT 若出现 NaN 会 mask 掉，不让 NaN 进 `Force6DEncoder`。

动作维是 TongBot 16D：`delta_action_dims: [7, 7, -1, -1]`（左7 | 右7 | 左爪 abs | 右爪 abs）。

## 预训练槽位是什么

π0.5 基座（`pi05_base`）prefix 只有两类槽，没有 DEPTH / FT：

| 槽 | 内容 | 长度 | 怎么对齐 |
|---|---|---|---|
| 图 1 | `base_0_rgb`（头/胸口，映射 `cam_mid`） | SigLIP So400m/14，224÷14=16 → **256** patch | 预训练三路相机之一 |
| 图 2 | `left_wrist_0_rgb`（`cam_left`） | 同上 256 | 同上 |
| 图 3 | `right_wrist_0_rgb`（`cam_right`） | 同上 256 | 同上 |
| 语言 | **一整段** Paligemma 文本，不是三个独立槽 | pad 到 `max_token_len=200` | 预训练语言模板 |

语言模板就是：

```text
Task: {指令}, State: {每维 0–255 的整数};
Action: 
```

`Task:` / `State:` / `Action:` 都是**同一段字符串里的字段标签**（冒号+空格），不是单独的特殊 token 类型。State 数字在这段里；Action 只有 cue，真动作在 suffix 的 50 个连续 token。

DEPTH / FT **没有**预训练槽。它们插在三路 RGB 和这段语言之间；标签写成同样的 `...: ` 形式，例如 `DEPTH WRIST LEFT: `。

## 语言 token（DEPTH / FT 标签）

YAML `data.append_modality_prompt` 控制要不要在 **DEPTH / FT 前面** 插 Paligemma 语言 token。RGB 已经和预训练三路相机对齐，**不加**标签，也不改 Task 文本。

开了以后 prefix：

```text
[RGB head][RGB left][RGB right]                 # 预训练图槽，无标签
[DEPTH WRIST LEFT: ][这一路 depth]               # 可选；HEAD/LEFT/RIGHT 三选一
[FT LEFT: ][左力][FT RIGHT: ][右力]               # 可选
[Task: pick banana, State: ...;
Action: ]
```

| 模态 key | 默认 token |
|---|---|
| `cam_mid_depth` | `DEPTH HEAD: ` |
| `cam_left_depth` | `DEPTH WRIST LEFT: ` |
| `cam_right_depth` | `DEPTH WRIST RIGHT: ` |
| `force6d.left` | `FT LEFT: ` |
| `force6d.right` | `FT RIGHT: ` |

DEPTH/FT 档 YAML 默认 `true`；只训 RGB 为 `false`。关掉则只加图 / Force6DEncoder。文案改 `data.modality_prompt_tags` 的 dict。tokenizer 会把 `_` 换成空格，标签请用空格。

## 开训

1. 填对应 YAML 的 `data.repo_id` / `data.assets.assets_dir` / `default_prompt` / `exp_name`。
2. 按该档的 `config_name` 算 norm stats：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_rgb_depth_head_ft \
    --dataset-dir /path/to/cigai20_ft_depth_lerobot
python scripts/train_from_yaml.py --config configs/train_pi05_rgb_depth_head_ft.yaml --dry-run
```

3. 点名服务器后再开训。没填 `repo_id` 不要开。
