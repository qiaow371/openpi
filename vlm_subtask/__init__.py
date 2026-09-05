"""π0.5 VLM high-level 拷贝：冻 SigLIP，只训/验短 subtask。

与 annotate_pipeline 解耦，改这里不影响夹爪切段，也不改 JAX 动作训练。
"""

__all__ = ["compose", "train_val"]
