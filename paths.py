# src/utils/paths.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

PROJECT_MARKERS = ("configs", "src")  # 看到这两个目录就认作项目根

def _looks_like_project_root(p: Path) -> bool:
    try:
        return (p / "configs").is_dir() and (p / "src").is_dir()
    except Exception:
        return False

def project_root_from(file: str, fallback_levels: int = 2) -> Path:
    """
    从给定 __file__ 开始向上找，直到遇到包含 'configs' 和 'src' 的目录，
    否则用 fallback: 往上 fallback_levels 层（默认 src/ 的上一层）。
    """
    p = Path(file).resolve()
    for cand in (p, *p.parents):
        if _looks_like_project_root(cand):
            return cand
    # 仍找不到时，退而求其次：向上 fallback_levels 层
    return p.parents[fallback_levels]

def data_dir(file: str) -> Path:
    """项目根下的 data/interim，若不存在不创建（由上层脚本决定）。"""
    return project_root_from(file) / "data" / "interim"

def outputs_dir(file: str) -> Path:
    """项目根下的 outputs。"""
    return project_root_from(file) / "outputs"