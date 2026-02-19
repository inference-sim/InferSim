"""
Implementation of should_use_tensor_core matching sglang's logic.

Reference: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py

This avoids sglang dependency while maintaining correct tensor core selection.
"""
import os
import torch


def should_use_tensor_core(
    kv_cache_dtype,
    num_attention_heads,
    num_kv_heads,
):
    """
    Determine whether to use tensor cores for attention computation.

    Based on sglang's implementation:
    - FP8: Always use tensor cores
    - FP16/BF16: Use tensor cores only if GQA group size >= 4
    - Other dtypes: Don't use tensor cores

    The GQA group size threshold exists because "a GQA group size of at
    least 4 is needed to efficiently use Tensor Cores" in FlashInfer.
    """
    # Environment variable override (priority 1)
    env_value = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")
    if env_value is not None:
        return env_value.lower() == "true"

    # Calculate GQA group size
    gqa_group_size = num_attention_heads // num_kv_heads

    # Float8 types: Always use tensor cores
    if kv_cache_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return True

    # Half-precision types: Use tensor cores only if group size >= 4
    if kv_cache_dtype in [torch.float16, torch.half, torch.bfloat16]:
        return gqa_group_size >= 4

    # Other dtypes: Don't use tensor cores
    return False
