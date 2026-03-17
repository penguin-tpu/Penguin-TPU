from __future__ import annotations

import math
from pathlib import Path

from penguin_model import (
    DEFAULT_PENGUIN_CORE_CONFIG,
    PI0_TOTAL_FLOPS,
    aggregate_workload_point,
    derive_roofline_metrics,
    format_report,
    format_pi0_workload_report,
    pi0_dominant_kernel_specs,
    pi0_workload_roofline_points,
    plot_pi0_workload_roofline,
    plot_roofline,
    representative_kernel_points,
)


def test_derive_roofline_metrics_matches_current_config() -> None:
    metrics = derive_roofline_metrics(DEFAULT_PENGUIN_CORE_CONFIG)

    assert metrics.dram_bandwidth_bytes_per_cycle == 2.0
    assert metrics.vmem_bandwidth_bytes_per_cycle == 16.0
    assert metrics.mxu_tile_ops == 524_288
    assert metrics.mxu_tile_bytes == 16_384
    assert metrics.mxu_peak_ops_per_cycle_per_mxu == 8_192.0
    assert metrics.mxu_peak_ops_per_cycle_total == 16_384.0
    assert metrics.vpu_tile_ops == 2_048
    assert metrics.vpu_unary_tile_bytes == 8_192
    assert metrics.vpu_peak_ops_per_cycle == 1_024.0
    assert metrics.mxu_dram_knee_ops_per_byte == 8_192.0
    assert metrics.mxu_vmem_knee_ops_per_byte == 1_024.0
    assert metrics.vpu_dram_knee_ops_per_byte == 512.0
    assert metrics.vpu_vmem_knee_ops_per_byte == 64.0


def test_representative_kernel_points_match_current_machine() -> None:
    points = {point.name: point for point in representative_kernel_points()}

    mxu = points["dual-mxu matmul tile"]
    assert mxu.ops == 1_048_576
    assert mxu.bytes_moved == 32_768
    assert math.isclose(mxu.arithmetic_intensity_ops_per_byte, 32.0)
    assert math.isclose(mxu.dram_ceiling_ops_per_cycle, 64.0)
    assert math.isclose(mxu.vmem_ceiling_ops_per_cycle, 512.0)
    assert mxu.dram_bound == "memory"
    assert mxu.vmem_bound == "memory"

    vpu = points["vpu unary tile"]
    assert vpu.ops == 2_048
    assert vpu.bytes_moved == 8_192
    assert vpu.arithmetic_intensity_ops_per_byte == 0.25
    assert vpu.dram_ceiling_ops_per_cycle == 0.5
    assert vpu.vmem_ceiling_ops_per_cycle == 4.0
    assert vpu.dram_bound == "memory"
    assert vpu.vmem_bound == "memory"


def test_format_report_includes_normalized_and_scaled_metrics() -> None:
    report = format_report(core_frequency_hz=500e6)

    assert "DRAM bandwidth: 2.00 B/cycle" in report
    assert "VMEM bandwidth: 16.00 B/cycle" in report
    assert "MXU peak throughput: 8192.00 ops/cycle per MXU, 16384.00 ops/cycle total" in report
    assert "VPU simple-op peak throughput: 1024.00 ops/cycle" in report
    assert "Core frequency: 500.00 MHz" in report
    assert "DRAM bandwidth: 1.000000 GB/s" in report
    assert "VMEM bandwidth: 8.000000 GB/s" in report
    assert "MXU total peak: 8192.000000 GOPS" in report
    assert "VPU simple-op peak: 512.000000 GOPS" in report


def test_plot_roofline_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "penguin_roofline.png"

    written_path = plot_roofline(output_path)

    assert written_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_pi0_dominant_kernel_specs_cover_the_reported_workload() -> None:
    specs = pi0_dominant_kernel_specs()

    assert len(specs) == 16
    assert specs[0].name == "vlm_mlp_gate_up"
    assert specs[0].total_flops == 1_971_389_988_864
    assert PI0_TOTAL_FLOPS == 4_354_614_038_072
    assert math.isclose(sum(spec.total_flops for spec in specs) / PI0_TOTAL_FLOPS, 0.9981956128549389)


def test_pi0_workload_roofline_points_match_current_machine() -> None:
    points = {point.name: point for point in pi0_workload_roofline_points()}

    top = points["vlm_mlp_gate_up"]
    assert top.calls == 36
    assert top.flops_per_call == 54_760_833_024
    assert top.dram_bytes_per_call == 61_964_288
    assert top.vmem_bytes_per_call == 886_046_720
    assert math.isclose(top.dram_intensity_ops_per_byte, 883.7482813326283)
    assert math.isclose(top.vmem_intensity_ops_per_byte, 61.80355029585799)
    assert math.isclose(top.dram_ceiling_ops_per_cycle, 1767.4965626652566)
    assert math.isclose(top.vmem_ceiling_ops_per_cycle, 988.8568047337278)
    assert top.dram_bound == "memory"
    assert top.vmem_bound == "memory"

    aggregate = aggregate_workload_point(list(points.values()))
    assert math.isclose(aggregate.share_total_flops, 0.9981956128549389)
    assert aggregate.dram_total_bytes == 10_199_343_744
    assert aggregate.vmem_total_bytes == 72_002_764_800
    assert math.isclose(aggregate.dram_intensity_ops_per_byte, 426.1800305570719)
    assert math.isclose(aggregate.vmem_intensity_ops_per_byte, 60.369301658816305)
    assert math.isclose(aggregate.dram_ceiling_ops_per_cycle, 852.3600611141438)
    assert math.isclose(aggregate.vmem_ceiling_ops_per_cycle, 965.9088265410609)


def test_format_pi0_workload_report_mentions_core_findings() -> None:
    report = format_pi0_workload_report(core_frequency_hz=500e6, limit=3)

    assert "Penguin PI0 workload roofline" in report
    assert "Prefix tokens: 816 = 3 x 256 image patches + 48 language tokens" in report
    assert "Suffix tokens: 51 = 1 state token + 50 action tokens" in report
    assert "Dominant kernel coverage: 73.57% of the 4,354,614,038,072 total FLOPs" in report
    assert "Aggregate dominant-kernel intensity: DRAM 924.00 ops/byte, VMEM 62.06 ops/byte" in report
    assert "Core frequency: 500.00 MHz" in report
    assert "vlm_mlp_gate_up:" in report


def test_plot_pi0_workload_roofline_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "penguin_pi0_roofline.png"

    written_path = plot_pi0_workload_roofline(output_path)

    assert written_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
