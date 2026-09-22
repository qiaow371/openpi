# 本机 π0.5：RGB / DEPTH / FT 消融

分支 `train/pi05-rgb-depth-ft-yaml`（从 `train/pi05-ft-depth` 拉出）。**不改** 现用 RGB-only `pi05_aloha`（`front/left/right` 旧数据集）。

## 选项

三路 RGB 始终进网（`cam_mid` / `cam_left` / `cam_right`）。DEPTH 和 FT 是独立开关；DEPTH 再选 HEAD 或 WRIST。

| 选项 | DEPTH | FT | `config_name` | YAML |
|---|---|---|---|---|
| 1. 只训练 RGB | 无 | 关 | `pi05_aloha_rgb` | [`configs/train_pi05_rgb.yaml`](../configs/train_pi05_rgb.yaml) |
| 2a. RGB + DEPTH HEAD | `cam_mid` | 关 | `pi05_aloha_rgb_depth_head` | [`configs/train_pi05_rgb_depth_head.yaml`](../configs/train_pi05_rgb_depth_head.yaml) |
| 2b. RGB + DEPTH WRIST | `cam_left` + `cam_right` | 关 | `pi05_aloha_rgb_depth_wrist` | [`configs/train_pi05_rgb_depth_wrist.yaml`](../configs/train_pi05_rgb_depth_wrist.yaml) |
| 3. RGB + FT | 无 | 开 | `pi05_aloha_rgb_ft` | [`configs/train_pi05_rgb_ft.yaml`](../configs/train_pi05_rgb_ft.yaml) |
| 4a. RGB + DEPTH HEAD + FT | `cam_mid` | 开 | `pi05_aloha_rgb_depth_head_ft` | [`configs/train_pi05_rgb_depth_head_ft.yaml`](../configs/train_pi05_rgb_depth_head_ft.yaml) |
| 4b. RGB + DEPTH WRIST + FT | `cam_left` + `cam_right` | 开 | `pi05_aloha_rgb_depth_wrist_ft` | [`configs/train_pi05_rgb_depth_wrist_ft.yaml`](../configs/train_pi05_rgb_depth_wrist_ft.yaml) |

HEAD = `cam_mid`（头/胸口）。WRIST = 左右腕。不要混用 `pi05_aloha`（那套是 `front/left/right`）。

旧预设 `pi05_aloha_ft_depth` 仍在：三路 depth 全开 + FT。新实验请用上表，不要再用 [`configs/train_pi05_ft_depth.yaml`](../configs/train_pi05_ft_depth.yaml)。

## 数据管线

Depth **不走** 独立 `DepthEncoder`，也不在采集侧 `cv2.resize`。
`rs.align` 之后 depth 已与 RGB 同 HxW；训练时复制成 3 通道灰度，并入 `images`，和 RGB 一起进 **SigLIP → Gemma**。

SigLIP So400m/14 的输入仍是 **224×224**（`ResizeImages`，`resize_with_pad`）。RGB 本来就是这条；这不是把 320×240 depth 硬拉到 640×480。

FT 仍走 `Force6DEncoder`。RGB-only / RGB+DEPTH 预设把 `use_force6d_encoder=False`，Repack 也不带 `force6d`。

DEPTH 档会挂 `LoadSidecarDepthPNGs(depth_keys=...)`，只读选中的相机 PNG，不会把三路都扫进来。

动作维是 TongBot 16D：`delta_action_dims: [7, 7, -1, -1]`（左7 | 右7 | 左爪 abs | 右爪 abs）。

## 开训

1. 填对应 YAML 的 `data.repo_id` / `data.assets.assets_dir` / `default_prompt` / `exp_name`。
2. 按该档的 `config_name` 算 norm stats：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_rgb_depth_head_ft \
    --dataset-dir /path/to/cigai20_ft_depth_lerobot
python scripts/train_from_yaml.py --config configs/train_pi05_rgb_depth_head_ft.yaml --dry-run
```

3. 点名服务器后再开训。没填 `repo_id` 不要开。
