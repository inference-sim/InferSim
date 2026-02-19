"""
Minimal implementation of should_use_tensor_core to avoid sglang dependency.
Based on flashinfer's tensor core usage guidelines.
"""
import torch


def should_use_tensor_core(
    kv_cache_dtype,
    num_attention_heads,
    num_kv_heads,
):
    """
    Determine whether to use tensor cores for attention computation.

    Tensor cores are beneficial when:
    1. Using FP16/BF16 dtype (not FP8 which doesn't use tensor cores the same way)
    2. For larger head counts (more parallelism)
    3. For GQA scenarios where num_attention_heads != num_kv_heads
    """
    # For FP8, tensor cores usage is handled differently
    if kv_cache_dtype == torch.float8_e4m3fn or kv_cache_dtype == torch.float8_e5m2:
        return False

    # For FP16/BF16, use tensor cores for better performance
    # Especially beneficial for GQA (grouped query attention)
    if kv_cache_dtype in [torch.float16, torch.bfloat16]:
        return True

    # Default to True for other cases
    return True
