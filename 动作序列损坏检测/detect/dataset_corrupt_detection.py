"""
动作序列损坏检测 - 数据加载适配模块

支持两种输入方式：
1. 目录扫描：扫描指定目录下所有 .npy 文件
2. 文件列表：从 txt 读取文件名列表（每行一个，支持相对路径）

复用 MotionMillion 的 mean/std 标准化逻辑，不修改原有 dataset/ 下任何文件。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# region 路径配置
_DETECT_DIR = Path(__file__).resolve().parent
_FEATURE_ROOT = _DETECT_DIR.parent
_REPO_ROOT = _FEATURE_ROOT.parent

for _p in (_REPO_ROOT, _FEATURE_ROOT, _DETECT_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)
# endregion


# region 数据集类
class CorruptDetectionDataset(Dataset):
    """
    用于损坏检测的运动数据集。

    与训练数据集的区别：不随机截取，每次返回完整序列（或按 unit_length 对齐后的序列），
    便于对整段动作进行损坏判定。
    """

    # region 初始化与配置
    def __init__(
        self,
        motion_dir: str,
        file_list: Optional[List[str]] = None,
        mean_std_dir: Optional[str] = None,
        motion_type: str = "vector_272",
        unit_length: int = 2,
        min_length: int = 64,
        recursive: bool = True,
    ):
        """
        Args:
            motion_dir: 运动文件根目录，如 dataset/MotionMillion/motion_data/vector_272
            file_list: 文件名列表。若为 None，则扫描 motion_dir 下所有 .npy
            mean_std_dir: mean/std 所在目录，默认 dataset/MotionMillion/mean_std/{motion_type}/
            motion_type: 运动类型，用于推断 mean_std 路径
            unit_length: 长度对齐单位（2^down_t），序列长度需为其整数倍
            min_length: 最小序列长度，过短则跳过
            recursive: 目录扫描时是否递归子目录
        """
        self.motion_dir = Path(motion_dir)
        self.unit_length = unit_length
        self.min_length = min_length

        # 加载 mean/std
        if mean_std_dir is None:
            data_root = _REPO_ROOT / "dataset" / "MotionMillion"
            mean_std_dir = data_root / "mean_std" / motion_type
        self.mean_std_dir = Path(mean_std_dir)
        self.mean = np.load(self.mean_std_dir / "mean.npy")
        self.std = np.load(self.mean_std_dir / "std.npy")

        # 构建文件列表
        if file_list is not None:
            self.file_paths = []
            for name in file_list:
                name = name.strip()
                if not name:
                    continue
                # 去掉 .npy 后缀（若有），再拼接
                base = name[:-4] if name.endswith(".npy") else name
                p = self.motion_dir / f"{base}.npy"
                if p.exists():
                    self.file_paths.append(p)
                else:
                    # 尝试作为绝对路径
                    alt = Path(name)
                    if alt.exists():
                        self.file_paths.append(alt)
        else:
            self.file_paths = self._scan_directory(recursive)

        self.file_paths = [p for p in self.file_paths if self._is_valid(p)]
    # endregion

    # region 内部方法
    def _scan_directory(self, recursive: bool) -> List[Path]:
        """扫描 motion_dir 下所有 .npy 文件"""
        paths = []
        if recursive:
            for p in self.motion_dir.rglob("*.npy"):
                paths.append(p)
        else:
            for p in self.motion_dir.glob("*.npy"):
                paths.append(p)
        return sorted(paths)

    def _is_valid(self, path: Path) -> bool:
        """检查文件是否有效（存在且长度足够）"""
        if not path.exists():
            return False
        try:
            motion = np.load(path)
            if motion.ndim != 2 or motion.shape[1] != self.mean.shape[0]:
                return False
            m_len = (len(motion) // self.unit_length) * self.unit_length
            return m_len >= self.min_length
        except Exception:
            return False
    # endregion

    # region 数据集接口
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.file_paths[idx]
        motion = np.load(path).astype(np.float32)
        m_len = (len(motion) // self.unit_length) * self.unit_length
        motion = motion[:m_len]
        motion = (motion - self.mean) / self.std
        try:
            name = str(path.relative_to(self.motion_dir))
        except ValueError:
            name = path.stem
        if name.endswith(".npy"):
            name = name[:-4]
        return torch.from_numpy(motion), name

    def inv_transform(self, data: np.ndarray) -> np.ndarray:
        """逆标准化，用于可视化等"""
        return data * self.std + self.mean
    # endregion
# endregion


# region 批处理辅助
def _collate_pad(batch: list) -> tuple:
    """
    自定义 collate：将可变长度序列 pad 到同一长度，便于 batch 处理。
    每个样本为 (motion, name)，motion 形状 (T, 272)。
    """
    motions, names = zip(*batch)
    max_len = max(m.shape[0] for m in motions)
    feat_dim = motions[0].shape[1]
    padded = []
    for m in motions:
        if m.shape[0] < max_len:
            pad = torch.zeros(max_len - m.shape[0], feat_dim, dtype=m.dtype)
            m = torch.cat([m, pad], dim=0)
        padded.append(m)
    return torch.stack(padded), list(names)
# endregion


# region DataLoader 工厂
def create_dataloader(
    motion_dir: str,
    file_list_path: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
    **dataset_kwargs,
) -> Tuple[DataLoader, CorruptDetectionDataset]:
    """
    创建用于损坏检测的 DataLoader。

    Args:
        motion_dir: 运动文件根目录
        file_list_path: 文件列表 txt 路径。若提供，则每行一个文件名（相对 motion_dir）
        batch_size: 批大小，检测时通常为 1
        num_workers: 加载进程数
        shuffle: 是否打乱
        **dataset_kwargs: 传给 CorruptDetectionDataset 的额外参数

    Returns:
        (dataloader, dataset)
    """
    file_list = None
    if file_list_path and os.path.exists(file_list_path):
        with open(file_list_path, "r", encoding="utf-8") as f:
            file_list = [line.strip() for line in f if line.strip()]

    dataset = CorruptDetectionDataset(motion_dir=motion_dir, file_list=file_list, **dataset_kwargs)
    # 使用自定义 collate 以支持可变长度序列的 batch
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=_collate_pad,
    )
    return loader, dataset


def iter_motion_files(
    motion_dir: str,
    file_list_path: Optional[str] = None,
    recursive: bool = True,
    **dataset_kwargs,
) -> Iterator[Tuple[np.ndarray, str]]:
    """
    轻量级迭代器：逐个 yield (motion, name)，不构建完整 Dataset。
    适用于流式处理或内存受限场景。
    """
    file_list = None
    if file_list_path and os.path.exists(file_list_path):
        with open(file_list_path, "r", encoding="utf-8") as f:
            file_list = [line.strip() for line in f if line.strip()]
    dataset = CorruptDetectionDataset(
        motion_dir=motion_dir,
        file_list=file_list,
        recursive=recursive,
        **dataset_kwargs,
    )
    for i in range(len(dataset)):
        motion, name = dataset[i]
        yield motion.numpy(), name
# endregion
