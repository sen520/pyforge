"""项目路径不随终端当前工作目录变化。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "hub"
