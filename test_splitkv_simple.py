#!/usr/bin/env python3

import torch
import sys
import os

# 添加当前目录到Python路径，以便导入flash_attn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flash_attn import flash_attn_func
from tests.test_flash_attn import attention_ref, attn_bias_from_alibi_slopes

def test_flash_attn_splitkv_simple():
    """
    简化版的 splitkv 测试，只测试前向传播，hdim=128, float16
    """
    device = "cuda"
    dtype = torch.float16
    
    # 测试配置
    seqlen_q, seqlen_k = 128, 256  # 选择一个中等大小的配置
    d = 128  # hdim = 128
    causal = False
    local = False
    alibi = False
    deterministic = False
    swap_sq_sk = False
    
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    
    # 设置随机种子
    torch.random.manual_seed(0)
    batch_size = 1
    nheads = 12
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    
    # 创建输入张量
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen_q, seqlen_k, causal=causal)
    else:
        alibi_slopes, attn_bias = None, None
    
    print(f"Testing splitkv forward pass with:")
    print(f"  seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  d={d}, dtype={dtype}")
    print(f"  causal={causal}, local={local}, alibi={alibi}")
    
    # Flash Attention forward pass
    try:
        out, lse, _ = flash_attn_func(
            q,
            k,
            v,
            0.0,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        print("✓ Flash Attention forward pass successful")
    except Exception as e:
        print(f"✗ Flash Attention forward pass failed: {e}")
        return False
    
    # 参考实现
    try:
        out_ref, attn_ref = attention_ref(
            q, k, v, None, None, attn_bias, 0.0, None, causal=causal, window_size=window_size
        )
        print("✓ Reference implementation successful")
    except Exception as e:
        print(f"✗ Reference implementation failed: {e}")
        return False
    
    # PyTorch 参考实现 (upcast=False, reorder_ops=True)
    try:
        out_pt, attn_pt = attention_ref(
            q,
            k,
            v,
            None,
            None,
            attn_bias,
            0.0,
            None,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )
        print("✓ PyTorch reference implementation successful")
    except Exception as e:
        print(f"✗ PyTorch reference implementation failed: {e}")
        return False
    
    # 数值精度检查
    max_diff = (out - out_ref).abs().max().item()
    mean_diff = (out - out_ref).abs().mean().item()
    pt_max_diff = (out_pt - out_ref).abs().max().item()
    pt_mean_diff = (out_pt - out_ref).abs().mean().item()
    
    print(f"\nNumerical Results:")
    print(f"  Flash Attention vs Reference:")
    print(f"    Max diff: {max_diff:.6f}")
    print(f"    Mean diff: {mean_diff:.6f}")
    print(f"  PyTorch vs Reference:")
    print(f"    Max diff: {pt_max_diff:.6f}")
    print(f"    Mean diff: {pt_mean_diff:.6f}")
    
    # 检查精度是否在合理范围内
    tolerance = 2 * pt_max_diff + 1e-5
    if max_diff <= tolerance:
        print(f"✓ Numerical accuracy check passed (max_diff <= {tolerance:.6f})")
        accuracy_ok = True
    else:
        print(f"✗ Numerical accuracy check failed (max_diff > {tolerance:.6f})")
        accuracy_ok = False
    
    # 总结
    print(f"\n{'='*50}")
    print(f"Test Summary (Forward Pass Only):")
    print(f"  Forward pass: {'✓ PASSED' if accuracy_ok else '✗ FAILED'}")
    print(f"  Overall: {'✓ PASSED' if accuracy_ok else '✗ FAILED'}")
    print(f"{'='*50}")
    
    return accuracy_ok

if __name__ == "__main__":
    print("Flash Attention Split-KV Test (Forward Pass Only, hdim=128, float16)")
    print("=" * 60)
    
    if torch.cuda.is_available():
        success = test_flash_attn_splitkv_simple()
        sys.exit(0 if success else 1)
    else:
        print("CUDA is not available!")
        sys.exit(1)