"""将上级 DroneDetect_V2 加入 import 路径，以便加载模型代码（不移动模型文件）。"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(_PKG_DIR)


def ensure_model_root() -> str:
    if MODEL_ROOT not in sys.path:
        sys.path.insert(0, MODEL_ROOT)
    return MODEL_ROOT


ensure_model_root()
