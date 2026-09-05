"""LeRobot v2.1 数据集后处理 Pipeline

模块:
  - structure_check: 数据集结构 & 元数据一致性检查
  - video_repair:    视频完整性检查 & 自动修复
  - data_stats:      数据集统计摘要
  - pipeline:        主流程编排

用法:
  python -m data_postprocessing.pipeline /path/to/dataset --fix --verify -j 4
"""

__version__ = "1.0.0"
