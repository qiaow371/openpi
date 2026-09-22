# `train/pi05-ft-depth`

从 `tongbot` 拉出，**不改** 现用 RGB-only `pi05_aloha`。

Depth **不走** 独立 `DepthEncoder`，也不在采集侧 `cv2.resize`。
`rs.align` 之后 depth 已与 RGB 同 HxW；训练时复制成 3 通道灰度，并入 `images`，和 RGB 一起进 **SigLIP → Gemma**。

SigLIP So400m/14 的输入仍是 **224×224**（`ResizeImages`，`resize_with_pad`）。RGB 本来就是这条；这不是把 320×240 depth 硬拉到 640×480。

FT 仍走 `Force6DEncoder`。

还需要：`LoadSidecarDepthPNGs`、cam_* Repack、`AlohaInputs` 转发 force6d。预设名 `pi05_aloha_ft_depth`。

开训前：填 `configs/train_pi05_ft_depth.yaml` 的 `data.repo_id`，跑 `compute_norm_stats.py --config-name pi05_aloha_ft_depth`，并点名服务器。
