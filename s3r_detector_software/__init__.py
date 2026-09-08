"""S3R / TSR-Open 检测软件（推理引擎 + GUI + 部署导出）。"""

from .bootstrap import MODEL_ROOT, ensure_model_root

__all__ = ['MODEL_ROOT', 'ensure_model_root']
