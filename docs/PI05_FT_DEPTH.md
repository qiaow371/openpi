# `train/pi05-ft-depth`

从 `tongbot` 拉出，**不改** 现用 RGB-only `pi05_aloha`。

只把 YAML 里 `use_depth_encoder` / `use_force6d_encoder` 改成 true **不够**。还必须：

1. 新预设 `pi05_aloha_ft_depth`（两个 encoder 打开）
2. Repack：`cam_*` RGB + `observation.depths.*` + `observation.force6d.{left,right}`
3. `LoadSidecarDepthPNGs`（LeRobot 不读旁路 uint16 PNG）
4. `ProcessDepths` / `ProcessForce6D` 挂进 `LeRobotAlohaDataConfig`
5. `AlohaInputs` 转发 `depths` / `force6d`
6. `ResizeDepths(224, 224)`（仅 encoder 打开时）

数采侧对应 DATA_COLLECT 分支 `cigai20-ft-depth`。

开训前：填 `configs/train_pi05_ft_depth.yaml` 的 `data.repo_id`，跑 `compute_norm_stats.py --config-name pi05_aloha_ft_depth`，并点名服务器。
