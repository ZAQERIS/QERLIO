import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.compression.quant_matrix import (
    QUANT_CATALOG,
    get_quant_profile,
    estimate_model_size_mb,
    calculate_compression_ratio,
    compute_layer_adaptive_mixed_precision,
    calculate_sink_aware_kv_cache
)
from server.local_api.server import app

client = TestClient(app)

def test_quant_catalog_completeness():
    assert len(QUANT_CATALOG) >= 28
    assert "Q4_K_M" in QUANT_CATALOG
    assert "IQ1_S" in QUANT_CATALOG
    assert "M3_GPTQ" in QUANT_CATALOG
    assert "FP8_E4M3" in QUANT_CATALOG
    assert "KV_Q4_0" in QUANT_CATALOG

def test_bits_per_weight_and_compression():
    q4km = get_quant_profile("Q4_K_M")
    assert q4km.bits_per_weight == 4.85
    ratio = calculate_compression_ratio("Q4_K_M")
    assert ratio > 3.0

    iq1s = get_quant_profile("IQ1_S")
    assert iq1s.bits_per_weight < 2.0
    ratio_iq1s = calculate_compression_ratio("IQ1_S")
    assert ratio_iq1s > 9.0

def test_layer_adaptive_hybrid_quantization():
    dahq = compute_layer_adaptive_mixed_precision(param_count_billions=7.0, base_quant="Q4_K_M")
    assert dahq["theoretical_estimate"] is True
    assert dahq["measured"] is False
    assert dahq["baseline_fp16_gb"] == 13.04
    assert dahq["hybrid_quant_gb"] < dahq["baseline_fp16_gb"]
    assert dahq["saved_vs_baseline_pct"] > 60.0
    assert dahq["reasoning_retention_score"] is None

def test_sink_aware_kv_cache_pruning():
    kv_opt = calculate_sink_aware_kv_cache(context_tokens=16384, kv_format="KV_Q4_0")
    assert kv_opt["theoretical_estimate"] is True
    assert kv_opt["measured"] is False
    assert kv_opt["standard_full_mb"] == 2048.0
    assert kv_opt["optimized_mb"] < 250.0
    assert kv_opt["saved_memory_pct"] > 85.0

def test_dashboard_endpoints():
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "QERLIO RUNTIME" in res_root.text
    assert "Universal Quantization Matrix" in res_root.text

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    for fake in ("65.1", "15.3", "3.85"):
        assert fake not in res_dash.text
    assert "no live measurement" in res_dash.text

def test_telemetry_hud_endpoint():
    res_hud = client.get("/v1/telemetry/hud")
    assert res_hud.status_code == 200
    d = res_hud.json()
    # No live inference is measured yet: metrics and host RAM must be null by default.
    assert d["measurement_status"] == "no_live_measurement"
    assert d["throughput_tps"] is None
    assert d["ttft_ms"] is None
    assert d["model_memory_gb"] is None
    assert d["ram_total_gb"] is None
    assert d["ram_available_gb"] is None
    assert "theoretical" in d["theoretical_estimates_note"].lower()
    assert d["quant_schemes_count"] >= 28
    assert d["adaptive_hybrid_7b"]["theoretical_estimate"] is True
    assert d["adaptive_hybrid_7b"]["measured"] is False
    assert d["sink_aware_16k"]["theoretical_estimate"] is True
    assert d["sink_aware_16k"]["measured"] is False
