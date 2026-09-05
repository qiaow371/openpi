#!/usr/bin/env python3
"""π0.5 VLM 独立入口，转发到 vlm_subtask 包。不要命名成 vlm_subtask.py，会和包抢名字。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm_subtask.__main__ import main

if __name__ == "__main__":
    main()
