from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

class QuantFamily(str, Enum):
    STANDARD_GGUF = "Standard GGUF"
    K_QUANTS = "k-quants"
    EXTREME_I_QUANTS = "i-quants (Sub-3bit)"
    ZAQERIS_RVQ = "ZAQERIS RVQ (Research)"
    HARDWARE_FP8 = "Hardware FP8 / INT8"
    KV_CACHE = "KV Cache Precision"

@dataclass
class QuantProfile:
    name: str
    family: QuantFamily
    bits_per_weight: float
    block_size: int
    # Perplexity impact is not measured by QERLIO yet; None means "no measurement".
    perplexity_delta: Optional[float]
    hardware_accelerated: bool

QUANT_CATALOG: Dict[str, QuantProfile] = {
    # Standard GGUF
    "F16": QuantProfile("F16", QuantFamily.STANDARD_GGUF, 16.0, 1, None, True),
    "BF16": QuantProfile("BF16", QuantFamily.STANDARD_GGUF, 16.0, 1, None, True),
    "Q8_0": QuantProfile("Q8_0", QuantFamily.STANDARD_GGUF, 8.50, 32, None, True),
    "Q5_1": QuantProfile("Q5_1", QuantFamily.STANDARD_GGUF, 6.00, 32, None, True),
    "Q5_0": QuantProfile("Q5_0", QuantFamily.STANDARD_GGUF, 5.50, 32, None, True),
    "Q4_1": QuantProfile("Q4_1", QuantFamily.STANDARD_GGUF, 5.00, 32, None, True),
    "Q4_0": QuantProfile("Q4_0", QuantFamily.STANDARD_GGUF, 4.50, 32, None, True),

    # k-quants
    "Q6_K": QuantProfile("Q6_K", QuantFamily.K_QUANTS, 6.56, 256, None, False),
    "Q5_K_M": QuantProfile("Q5_K_M", QuantFamily.K_QUANTS, 5.69, 256, None, False),
    "Q5_K_S": QuantProfile("Q5_K_S", QuantFamily.K_QUANTS, 5.50, 256, None, False),
    "Q4_K_M": QuantProfile("Q4_K_M", QuantFamily.K_QUANTS, 4.85, 256, None, False),
    "Q4_K_S": QuantProfile("Q4_K_S", QuantFamily.K_QUANTS, 4.50, 256, None, False),
    "Q3_K_L": QuantProfile("Q3_K_L", QuantFamily.K_QUANTS, 4.27, 256, None, False),
    "Q3_K_M": QuantProfile("Q3_K_M", QuantFamily.K_QUANTS, 3.91, 256, None, False),
    "Q3_K_S": QuantProfile("Q3_K_S", QuantFamily.K_QUANTS, 3.44, 256, None, False),
    "Q2_K": QuantProfile("Q2_K", QuantFamily.K_QUANTS, 2.56, 256, None, False),

    # Extreme i-quants (Sub-3bit)
    "IQ4_NL": QuantProfile("IQ4_NL", QuantFamily.EXTREME_I_QUANTS, 4.50, 32, None, False),
    "IQ4_XS": QuantProfile("IQ4_XS", QuantFamily.EXTREME_I_QUANTS, 4.25, 256, None, False),
    "IQ3_S": QuantProfile("IQ3_S", QuantFamily.EXTREME_I_QUANTS, 3.44, 256, None, False),
    "IQ3_XXS": QuantProfile("IQ3_XXS", QuantFamily.EXTREME_I_QUANTS, 3.06, 256, None, False),
    "IQ2_S": QuantProfile("IQ2_S", QuantFamily.EXTREME_I_QUANTS, 2.50, 256, None, False),
    "IQ2_XS": QuantProfile("IQ2_XS", QuantFamily.EXTREME_I_QUANTS, 2.31, 256, None, False),
    "IQ2_XXS": QuantProfile("IQ2_XXS", QuantFamily.EXTREME_I_QUANTS, 2.06, 256, None, False),
    "IQ1_M": QuantProfile("IQ1_M", QuantFamily.EXTREME_I_QUANTS, 1.75, 256, None, False),
    "IQ1_S": QuantProfile("IQ1_S", QuantFamily.EXTREME_I_QUANTS, 1.56, 256, None, False),

    # ZAQERIS RVQ
    "M3_GPTQ": QuantProfile("M3_GPTQ", QuantFamily.ZAQERIS_RVQ, 3.16, 8, None, True),

    # Hardware FP8
    "FP8_E4M3": QuantProfile("FP8_E4M3", QuantFamily.HARDWARE_FP8, 8.00, 1, None, True),
    "FP8_E5M2": QuantProfile("FP8_E5M2", QuantFamily.HARDWARE_FP8, 8.00, 1, None, True),

    # KV Cache
    "KV_Q8_0": QuantProfile("KV_Q8_0", QuantFamily.KV_CACHE, 8.00, 32, None, True),
    "KV_Q4_0": QuantProfile("KV_Q4_0", QuantFamily.KV_CACHE, 4.00, 32, None, True)
}

def get_quant_profile(name: str) -> Optional[QuantProfile]:
    return QUANT_CATALOG.get(name.upper())

def estimate_model_size_mb(param_count_billions: float, quant_name: str) -> float:
    p = get_quant_profile(quant_name)
    if not p:
        p = QUANT_CATALOG["Q4_K_M"]
    total_bits = param_count_billions * 1e9 * p.bits_per_weight
    return round(total_bits / (8 * 1024 * 1024), 2)

def calculate_compression_ratio(quant_name: str) -> float:
    p = get_quant_profile(quant_name)
    if not p:
        return 1.0
    return round(16.0 / p.bits_per_weight, 2)

def compute_layer_adaptive_mixed_precision(
    param_count_billions: float = 7.0,
    base_quant: str = "Q4_K_M"
) -> Dict[str, Any]:
    """
    Theoretical byte-size formula estimate for layer-adaptive mixed precision
    (8.5 bpw for 15% embedding/attention parameters, 3.91 bpw for 85% MLP parameters).
    Theoretical estimate only -- not measured on live model weights.
    """
    total_params = param_count_billions * 1e9
    critical_params = total_params * 0.15
    mlp_params = total_params * 0.85

    baseline_bytes = (total_params * 16.0) / 8.0
    uniform_bytes = (total_params * QUANT_CATALOG[base_quant].bits_per_weight) / 8.0

    # Mixed precision: Q8 for critical, Q3_K_M for MLP
    hybrid_bytes = ((critical_params * 8.5) + (mlp_params * 3.91)) / 8.0

    saved_vs_baseline_pct = ((baseline_bytes - hybrid_bytes) / baseline_bytes) * 100.0
    saved_vs_uniform_pct = ((uniform_bytes - hybrid_bytes) / uniform_bytes) * 100.0

    return {
        "theoretical_estimate": True,
        "measured": False,
        "baseline_fp16_gb": round(baseline_bytes / (1024**3), 2),
        "uniform_quant_gb": round(uniform_bytes / (1024**3), 2),
        "hybrid_quant_gb": round(hybrid_bytes / (1024**3), 2),
        "saved_vs_baseline_pct": round(saved_vs_baseline_pct, 1),
        "saved_vs_uniform_pct": round(saved_vs_uniform_pct, 1),
        "reasoning_retention_score": None
    }

def calculate_sink_aware_kv_cache(
    layers: int = 32,
    kv_heads: int = 8,
    head_dim: int = 128,
    context_tokens: int = 16384,
    sink_tokens: int = 4,
    heavy_hitter_ratio: float = 0.20,
    kv_format: str = "KV_Q4_0"
) -> Dict[str, Any]:
    """
    Theoretical byte-size formula estimate for a sink-token + heavy-hitter KV cache layout.
    Theoretical estimate only -- not measured on live model execution.
    """
    bytes_per_elem = 0.5 if kv_format == "KV_Q4_0" else (1.0 if kv_format == "KV_Q8_0" else 2.0)
    
    # Standard full uncompressed cache (FP16 = 2.0 bytes)
    standard_full_bytes = 2 * layers * kv_heads * head_dim * context_tokens * 2.0
    
    # Pruned tokens count: sink tokens + top attention tokens
    retained_tokens = sink_tokens + int(context_tokens * heavy_hitter_ratio)
    optimized_bytes = 2 * layers * kv_heads * head_dim * retained_tokens * bytes_per_elem
    
    saved_pct = ((standard_full_bytes - optimized_bytes) / standard_full_bytes) * 100.0
    
    return {
        "theoretical_estimate": True,
        "measured": False,
        "context_tokens": context_tokens,
        "standard_full_mb": round(standard_full_bytes / (1024**2), 1),
        "optimized_mb": round(optimized_bytes / (1024**2), 1),
        "saved_memory_pct": round(saved_pct, 1),
        "kv_format": kv_format
    }
