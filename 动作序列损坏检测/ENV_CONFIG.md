# 动作序列损坏检测 - 环境与依赖配置

## 1.1 预训练模型路径

| 配置项 | 路径/值 | 说明 |
|--------|---------|------|
| VQ-VAE (FSQ) 检查点 | `checkpoints/pretrained_models/fsq_net_6000000.pth` | 与 `train_tokenizer.sh` 训练产出一致 |
| 备用路径 | `results/output/FSQ_96len/.../net_6000000.pth` | 若自训练则使用对应输出目录 |

**注意**：若 `checkpoints/` 被 gitignore，需确保该文件已下载或从训练结果复制到上述路径。

## 1.2 数据格式

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 数据集 | MotionMillion | `dataname=motionmillion` |
| 运动类型 | vector_272 | `motion_type=vector_272` |
| 特征维度 | 272 | 与 vector_272 一致 |
| 文件格式 | `.npy` | 形状 `(T, 272)`，T 为帧数 |
| 标准化 | Z-score | `(motion - mean) / std` |
| mean 路径 | `dataset/MotionMillion/mean_std/vector_272/mean.npy` | |
| std 路径 | `dataset/MotionMillion/mean_std/vector_272/std.npy` | |

**窗口与下采样**：
- `window_size`: 96（训练时）
- `unit_length`: 2^down_t = 2
- `down_t`: 1
- 输入长度需为 `unit_length` 的整数倍

## 1.3 模型超参数（与训练一致）

| 参数 | 值 | 说明 |
|------|-----|------|
| dataname | motionmillion | |
| quantizer | FSQ | |
| nb_code | 65536 | 码本大小 |
| down_t | 1 | 时间下采样 |
| motion_type | vector_272 | |
| use_patcher | True | |
| patch_size | 1 | |
| patch_method | haar | |
| vq_norm | LN | |
| kernel_size | 3 | |
| depth | 3 | |

## 1.4 目录结构

```
MotionMillion-Codes/
├── 动作序列损坏检测/
│   ├── ENV_CONFIG.md           # 本配置文档
│   ├── 动作序列损坏检测方案_*.plan.md
│   ├── detect_corrupt_utils.py  # 核心检测逻辑
│   └── ...
├── checkpoints/pretrained_models/
│   └── fsq_net_6000000.pth
├── dataset/MotionMillion/
│   ├── mean_std/vector_272/
│   │   ├── mean.npy
│   │   └── std.npy
│   └── motion_data/vector_272/
└── ...
```
