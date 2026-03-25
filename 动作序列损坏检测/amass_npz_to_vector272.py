"""
将 AMASS / StableMotion 原始目录下的 .npz（poses/trans）转为 MotionMillion vector_272 的 .npy 副本。

依赖：
  - 本仓库根在 PYTHONPATH（与 generate_corrupt_data 相同）
  - StableMotion 工程根目录（--stablemotion-root），用于 data_loaders.amasstools
  - SMPL+H 资源位于 StableMotion 的 data_loaders/amasstools/deps（或 --smpl-deps）

不修改、不覆盖输入目录中的任何文件。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, cast

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tqdm import tqdm

from utils.motion_vector272_encode import (
    encode_world_joints_and_rotations_to_vector272,
    roundtrip_self_test_from_vector272,
    smpl22_fk_global_rotations,
)


def _ensure_stablemotion_on_path(stablemotion_root: str) -> None:
    root = str(Path(stablemotion_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _collect_npz_files(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(input_dir.rglob("*.npz"))
    return sorted(input_dir.glob("*.npz"))


def _resample_amass_npz(
    data: dict,
    old_fps: float,
    intermediate_fps: float,
    target_fps: float,
    interpolate_fps_poses,
    interpolate_fps_trans,
) -> tuple[np.ndarray, np.ndarray]:
    poses = torch.from_numpy(data["poses"]).float()
    trans = torch.from_numpy(data["trans"]).float()
    if poses.shape[-1] > 66:
        poses = poses[:, :66]
    elif poses.shape[-1] != 66:
        raise ValueError(f"poses last dim must be 66 or >=156, got {poses.shape[-1]}")

    try:
        if abs(old_fps - intermediate_fps) > 1e-6:
            poses = interpolate_fps_poses(poses, old_fps, intermediate_fps)
            trans = interpolate_fps_trans(trans, old_fps, intermediate_fps)

        if abs(float(intermediate_fps) - float(target_fps)) > 1e-6:
            poses = interpolate_fps_poses(poses, intermediate_fps, target_fps)
            trans = interpolate_fps_trans(trans, intermediate_fps, target_fps)
    except RuntimeError:
        if len(trans) == 1:
            poses = poses
            trans = trans
        else:
            raise

    return poses.numpy().astype(np.float64), trans.numpy().astype(np.float64)


def _run_self_test() -> None:
    from utils.motion_vector272_encode import roundtrip_self_test_synthetic

    m = roundtrip_self_test_synthetic()
    print("synthetic roundtrip:", m)


def main() -> None:
    parser = argparse.ArgumentParser(description="AMASS npz → MotionMillion vector_272 npy")
    parser.add_argument("--input-dir", type=str, required=True, help="含 .npz 的根目录")
    parser.add_argument("--output-dir", type=str, required=True, help="输出 .npy 根目录（镜像相对路径）")
    parser.add_argument(
        "--stablemotion-root",
        type=str,
        required=True,
        help="StableMotion-改进模型训练 工程根目录（含 data_loaders 包）",
    )
    parser.add_argument(
        "--smpl-deps",
        type=str,
        default="",
        help="SMPL+H 模型目录（内含 SMPLH 权重）；默认 .../amasstools/deps/smplh",
    )
    parser.add_argument("--target-fps", type=float, default=30.0, help="输出序列帧率（默认 30，与 MotionMillion eval 一致）")
    parser.add_argument(
        "--intermediate-fps",
        type=float,
        default=20.0,
        help="先重采样到此帧率（与 StableMotion fix_fps 一致），再插值到 target-fps",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female", "gendered"])
    parser.add_argument("--batch-size", type=int, default=4096, help="SMPLH 分块大小")
    parser.add_argument("--use-betas", action="store_true", help="若 npz 含 betas 则传入 SMPLH")
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", action="store_true", help="仅扫描当前目录")
    parser.add_argument("--min-length", type=int, default=16, help="短于此帧数的序列跳过")
    parser.add_argument("--self-test", action="store_true", help="仅运行编码往返自测后退出")
    parser.add_argument("--verbose-roundtrip", action="store_true", help="对首条成功样本打印 roundtrip 误差（需 torch）")
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return

    _ensure_stablemotion_on_path(args.stablemotion_root)
    # 包在 StableMotion 仓库根下；IDE 解析见仓库根 pyrightconfig.json / .vscode settings extraPaths
    from data_loaders.amasstools.fix_fps import interpolate_fps_poses, interpolate_fps_trans
    from data_loaders.amasstools.smplh_layer import load_smplh_gender

    smpl_deps = args.smpl_deps or os.path.join(
        args.stablemotion_root, "data_loaders", "amasstools", "deps", "smplh"
    )
    if not os.path.isdir(smpl_deps):
        raise FileNotFoundError(f"SMPL deps 目录不存在: {smpl_deps}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，改用 cpu")
        device = torch.device("cpu")

    jointstype = "smpljoints"
    smplh = load_smplh_gender(
        args.gender,
        smpl_deps,
        jointstype,
        args.batch_size,
        device,
        input_pose_rep="axisangle",
    )

    input_path = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    recursive = args.recursive and not args.no_recursive

    if not input_path.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_path}")

    files = _collect_npz_files(input_path, recursive)
    if not files:
        print("未找到 .npz 文件")
        return

    first_verbose = args.verbose_roundtrip

    for fp in tqdm(files, desc="npz→vector_272"):
        rel = fp.relative_to(input_path)
        out_path = output_root / rel.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            raw = np.load(str(fp), allow_pickle=True)
            data = {k: raw[k] for k in raw.files}
            old_fps = float(data["mocap_framerate"])
        except Exception as e:
            print(f"跳过（读取失败）{fp}: {e}")
            continue

        try:
            poses_np, trans_np = _resample_amass_npz(
                data,
                old_fps,
                args.intermediate_fps,
                args.target_fps,
                interpolate_fps_poses,
                interpolate_fps_trans,
            )
        except Exception as e:
            print(f"跳过（重采样失败）{fp}: {e}")
            continue

        tlen = poses_np.shape[0]
        if tlen < args.min_length:
            continue

        poses_t = torch.from_numpy(poses_np.astype(np.float32)).to(device)
        trans_t = torch.from_numpy(trans_np.astype(np.float32)).to(device)

        betas_t: Optional[torch.Tensor] = None
        if args.use_betas and "betas" in data and data["betas"] is not None:
            bet = torch.from_numpy(np.asarray(data["betas"])).float().to(device)
            if bet.dim() == 1:
                bet = bet.unsqueeze(0).expand(tlen, -1)
            betas_t = bet

        try:
            if args.gender == "gendered":
                if "gender" not in data:
                    print(f"跳过（gendered 但无 gender 字段）{fp}")
                    continue
                g = data["gender"]
                if isinstance(g, bytes):
                    g = g.decode("utf-8", errors="replace")
                else:
                    g = str(g)
                layer = smplh[g]
                joints = layer(poses_t, trans_t, betas_t)
            else:
                single = cast(Callable[..., torch.Tensor], smplh)
                joints = single(poses_t, trans_t, betas_t)
        except Exception as e:
            print(f"跳过（SMPLH 失败）{fp}: {e}")
            continue

        joints_np = joints.detach().float().cpu().numpy()
        if joints_np.shape[-2] < 22:
            print(f"跳过（关节数不足 22）{fp}: shape={joints_np.shape}")
            continue
        W = joints_np[:, :22, :].astype(np.float64)
        R_glob = smpl22_fk_global_rotations(poses_np)
        vec = encode_world_joints_and_rotations_to_vector272(W, R_glob)
        np.save(str(out_path), vec.astype(np.float32))

        if first_verbose:
            first_verbose = False
            try:
                m = roundtrip_self_test_from_vector272(vec)
                print(f"首条样本 roundtrip: {fp.name} -> {m}")
            except Exception as e:
                print(f"roundtrip 检查失败: {e}")


if __name__ == "__main__":
    main()
