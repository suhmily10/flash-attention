#!/usr/bin/env python3

import torch
import sys
import os

# 添加当前目录到Python路径，以便导入flash_attn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flash_attn import flash_attn_with_kvcache
from tests.test_flash_attn import attention_ref

def test_flash_attn_splitkv_real():
    """
    真正的 splitkv 测试，使用 flash_attn_with_kvcache 和 num_splits 参数
    """
    device = "cuda"
    dtype = torch.float16
    
    # 测试配置
    batch_size = 2
    seqlen_q = 64
    seqlen_k = 2048  # 使用较长的序列长度来体现 splitkv 的优势
    nheads = 32
    nheads_kv = 2  # 测试 GQA (Grouped Query Attention)
    d = 128
    
    # 设置随机种子
    torch.random.manual_seed(0)
    
    # 创建输入张量
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k_cache = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    
    # cache_seqlens 指定每个 batch 中 KV cache 的有效长度
    cache_seqlens = torch.tensor([seqlen_k] * batch_size, dtype=torch.int32, device=device)
    
    print(f"Testing real splitkv interface with:")
    print(f"  batch_size={batch_size}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  nheads={nheads}, nheads_kv={nheads_kv}, d={d}")
    print(f"  dtype={dtype}")
    
    # 测试不同的 num_splits 值
    results = {}
    
    # 1. 测试 num_splits=1 (无分割)
    print(f"\n{'='*60}")
    print("Testing num_splits=1 (no splitting):")
    try:
        out_nosplit, lse_nosplit = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            num_splits=1,
            return_softmax_lse=True
        )
        print("✓ num_splits=1 successful")
        results['nosplit'] = (out_nosplit, lse_nosplit)
    except Exception as e:
        print(f"✗ num_splits=1 failed: {e}")
        return False
    
    # 2. 测试 num_splits=4 (手动分割)
    print(f"\n{'='*60}")
    print("Testing num_splits=4 (manual splitting):")
    try:
        out_split4, lse_split4 = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            num_splits=4,
            return_softmax_lse=True
        )
        print("✓ num_splits=4 successful")
        results['split4'] = (out_split4, lse_split4)
    except Exception as e:
        print(f"✗ num_splits=4 failed: {e}")
        return False
    
    # 参考实现 (用于数值验证)
    print(f"\n{'='*60}")
    print("Computing reference implementation:")
    try:
        # 将 k_cache 和 v_cache 扩展到匹配 q 的 head 数量 (GQA)
        k_cache_expanded = k_cache.repeat_interleave(nheads // nheads_kv, dim=2)
        v_cache_expanded = v_cache.repeat_interleave(nheads // nheads_kv, dim=2)
        
        out_ref, _ = attention_ref(
            q, k_cache_expanded, v_cache_expanded,
            None, None, None, 0.0, None,
            causal=False, window_size=(-1, -1)
        )
        print("✓ Reference implementation successful")
        results['reference'] = out_ref
    except Exception as e:
        print(f"✗ Reference implementation failed: {e}")
        return False
    
    # 数值精度检查
    print(f"\n{'='*60}")
    print("Numerical accuracy checks:")
    
    accuracy_ok = True
    
    # 检查不同 splitkv 设置之间的一致性
    for name1, name2 in [('nosplit', 'split4')]:
        if name1 in results and name2 in results:
            out1, lse1 = results[name1]
            out2, lse2 = results[name2]
            
            max_diff_out = (out1 - out2).abs().max().item()
            mean_diff_out = (out1 - out2).abs().mean().item()
            max_diff_lse = (lse1 - lse2).abs().max().item()
            mean_diff_lse = (lse1 - lse2).abs().mean().item()
            
            print(f"  {name1} vs {name2}:")
            print(f"    Output max diff: {max_diff_out:.8f}")
            print(f"    Output mean diff: {mean_diff_out:.8f}")
            print(f"    LSE max diff: {max_diff_lse:.8f}")
            print(f"    LSE mean diff: {mean_diff_lse:.8f}")
            
            # splitkv 之间应该完全一致
            if max_diff_out > 1e-3 or max_diff_lse > 1e-3:
                print(f"    ✗ Difference too large between {name1} and {name2}")
                accuracy_ok = False
            else:
                print(f"    ✓ Consistent results between {name1} and {name2}")
    
    # 与参考实现比较
    if 'reference' in results and 'nosplit' in results:
        out_nosplit, _ = results['nosplit']
        out_ref = results['reference']
        
        max_diff_ref = (out_nosplit - out_ref).abs().max().item()
        mean_diff_ref = (out_nosplit - out_ref).abs().mean().item()
        
        print(f"  Flash Attention vs Reference:")
        print(f"    Max diff: {max_diff_ref:.8f}")
        print(f"    Mean diff: {mean_diff_ref:.8f}")
        
        # 与参考实现的误差应该在合理范围内
        if max_diff_ref > 2e-3:  # 允许较大的数值误差，因为使用了 float16
            print(f"    ✗ Difference too large vs reference")
            accuracy_ok = False
        else:
            print(f"    ✓ Acceptable difference vs reference")
    
    # 总结
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"  SplitKV interface: ✓ WORKING")
    print(f"  Multiple split configurations: ✓ TESTED")
    print(f"  Numerical accuracy: {'✓ PASSED' if accuracy_ok else '✗ FAILED'}")
    print(f"  Overall: {'✓ PASSED' if accuracy_ok else '✗ FAILED'}")
    print(f"{'='*60}")
    
    return accuracy_ok

if __name__ == "__main__":
    print("Flash Attention Split-KV Test (Real Interface)")
    print("=" * 60)
    
    if torch.cuda.is_available():
        success = test_flash_attn_splitkv_real()
        sys.exit(0 if success else 1)
    else:
        print("CUDA is not available!")
        sys.exit(1)