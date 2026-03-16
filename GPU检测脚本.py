#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch GPU 检测脚本
用于检查当前 PyTorch 是否为 GPU 版本，以及是否能正确识别显卡
"""

import sys
import io

# 解决 Windows 控制台中文/特殊字符编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def main():
    print("=" * 60)
    print("PyTorch GPU 环境检测")
    print("=" * 60)
    
    # 1. 检查 PyTorch 是否安装
    try:
        import torch
        print(f"\n[OK] PyTorch 已安装")
        print(f"  版本: {torch.__version__}")
    except ImportError as e:
        print(f"\n[FAIL] PyTorch 未安装: {e}")
        sys.exit(1)
    
    # 2. 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA 可用: {'[OK] 是 (GPU 版本)' if cuda_available else '[FAIL] 否 (CPU 版本)'}")
    
    if cuda_available:
        # 3. CUDA 版本信息
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
        
        # 4. 显卡数量与信息
        print(f"\n检测到 {torch.cuda.device_count()} 张显卡:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - 显存: {props.total_memory / 1024**3:.2f} GB")
            print(f"    - 计算能力: {props.major}.{props.minor}")
        
        # 5. 当前默认设备
        print(f"\n当前默认设备: cuda:{torch.cuda.current_device()}")
        
        # 6. 简单测试：在 GPU 上创建张量
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x @ y
            print(f"\n[OK] GPU 计算测试通过 (矩阵乘法)")
        except Exception as e:
            print(f"\n[FAIL] GPU 计算测试失败: {e}")
    else:
        print("\n提示: 如需使用 GPU，请安装 CUDA 版本的 PyTorch:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  (根据你的 CUDA 版本选择 cu118 或 cu121 等)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
