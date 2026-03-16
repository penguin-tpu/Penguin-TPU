"""Spreadsheet-style roofline model for the current Penguin design."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .core_config import DEFAULT_PENGUIN_CORE_CONFIG, PenguinCoreConfig


@dataclass(frozen=True, slots=True)
class RooflineMetrics:
    """Derived hardware limits for the current core configuration."""

    dram_bandwidth_bytes_per_cycle: float
    vmem_bandwidth_bytes_per_cycle: float
    mxu_tile_ops: int
    mxu_peak_ops_per_cycle_per_mxu: float
    mxu_peak_ops_per_cycle_total: float
    vpu_tile_ops: int
    vpu_peak_ops_per_cycle: float
    mxu_dram_knee_ops_per_byte: float
    mxu_vmem_knee_ops_per_byte: float
    vpu_dram_knee_ops_per_byte: float
    vpu_vmem_knee_ops_per_byte: float
    mxu_tile_bytes: int
    vpu_unary_tile_bytes: int


@dataclass(frozen=True, slots=True)
class RooflinePoint:
    """One representative kernel projected onto the roofline."""

    name: str
    ops: int
    bytes_moved: int
    arithmetic_intensity_ops_per_byte: float
    compute_peak_ops_per_cycle: float
    dram_ceiling_ops_per_cycle: float
    vmem_ceiling_ops_per_cycle: float
    dram_bound: str
    vmem_bound: str


MatrixKernelOperator = Literal["mm", "addmm", "bmm"]

PI0_TOTAL_FLOPS = 4_354_614_038_072


@dataclass(frozen=True, slots=True)
class WorkloadKernelSpec:
    """One dominant PI0 kernel extracted from the external FLOP report."""

    name: str
    stage: str
    operator: MatrixKernelOperator
    calls: int
    total_flops: int
    shape: tuple[int, ...]
    notes: str
    bias: bool = False


@dataclass(frozen=True, slots=True)
class WorkloadRooflinePoint:
    """Roofline projection for one workload kernel under DRAM and VMEM traffic models."""

    name: str
    stage: str
    operator: MatrixKernelOperator
    calls: int
    total_flops: int
    flops_per_call: int
    share_total_flops: float
    shape: tuple[int, ...]
    dram_bytes_per_call: int
    vmem_bytes_per_call: int
    dram_total_bytes: int
    vmem_total_bytes: int
    dram_intensity_ops_per_byte: float
    vmem_intensity_ops_per_byte: float
    compute_peak_ops_per_cycle: float
    dram_ceiling_ops_per_cycle: float
    vmem_ceiling_ops_per_cycle: float
    dram_bound: str
    vmem_bound: str
    notes: str


PI0_DOMINANT_KERNEL_SPECS = (
    WorkloadKernelSpec(
        name="vlm_mlp_gate_up",
        stage="Gemma prefix",
        operator="mm",
        calls=36,
        total_flops=1_971_389_988_864,
        shape=(816, 2048, 16384),
        notes="Two 2048->16384 MLP projections across the 18 Gemma prefix layers.",
    ),
    WorkloadKernelSpec(
        name="vlm_mlp_down",
        stage="Gemma prefix",
        operator="mm",
        calls=18,
        total_flops=985_694_994_432,
        shape=(816, 16384, 2048),
        notes="16384->2048 Gemma prefix MLP down projection.",
    ),
    WorkloadKernelSpec(
        name="vlm_attention_q_o",
        stage="Gemma prefix",
        operator="mm",
        calls=36,
        total_flops=246_423_748_608,
        shape=(816, 2048, 2048),
        notes="2048-wide Gemma prefix attention projections and output projection cluster.",
    ),
    WorkloadKernelSpec(
        name="siglip_attention_proj",
        stage="SigLIP vision",
        operator="addmm",
        calls=324,
        total_flops=220_150_628_352,
        shape=(256, 1152, 1152),
        bias=True,
        notes="SigLIP 1152-wide projection-heavy layers reported as addmm with bias.",
    ),
    WorkloadKernelSpec(
        name="siglip_mlp_fc1",
        stage="SigLIP vision",
        operator="addmm",
        calls=81,
        total_flops=205_626_802_176,
        shape=(256, 1152, 4304),
        bias=True,
        notes="SigLIP MLP expansion projection 1152->4304 with bias.",
    ),
    WorkloadKernelSpec(
        name="siglip_mlp_fc2",
        stage="SigLIP vision",
        operator="addmm",
        calls=81,
        total_flops=205_626_802_176,
        shape=(256, 4304, 1152),
        bias=True,
        notes="SigLIP MLP contraction projection 4304->1152 with bias.",
    ),
    WorkloadKernelSpec(
        name="expert_mlp_up",
        stage="Action expert",
        operator="mm",
        calls=360,
        total_flops=154_014_842_880,
        shape=(51, 1024, 4096),
        notes="Action-expert MLP expansion projection 1024->4096.",
    ),
    WorkloadKernelSpec(
        name="expert_mlp_down",
        stage="Action expert",
        operator="mm",
        calls=180,
        total_flops=77_007_421_440,
        shape=(51, 4096, 1024),
        notes="Action-expert MLP contraction projection 4096->1024.",
    ),
    WorkloadKernelSpec(
        name="vlm_attention_scores",
        stage="Gemma prefix",
        operator="bmm",
        calls=18,
        total_flops=49_092_231_168,
        shape=(8, 816, 256, 816),
        notes="Prefix attention score matrix multiply over 8 heads and 816 prefix tokens.",
    ),
    WorkloadKernelSpec(
        name="vlm_attention_context",
        stage="Gemma prefix",
        operator="bmm",
        calls=18,
        total_flops=49_092_231_168,
        shape=(8, 816, 816, 256),
        notes="Prefix attention context projection over 8 heads and 816 prefix tokens.",
    ),
    WorkloadKernelSpec(
        name="expert_attention_q",
        stage="Action expert",
        operator="mm",
        calls=180,
        total_flops=38_503_710_720,
        shape=(51, 1024, 2048),
        notes="Action-expert attention query projection 1024->2048.",
    ),
    WorkloadKernelSpec(
        name="expert_attention_o",
        stage="Action expert",
        operator="mm",
        calls=180,
        total_flops=38_503_710_720,
        shape=(51, 2048, 1024),
        notes="Action-expert attention output projection 2048->1024.",
    ),
    WorkloadKernelSpec(
        name="expert_attention_scores",
        stage="Action expert",
        operator="bmm",
        calls=180,
        total_flops=32_600_309_760,
        shape=(8, 51, 256, 867),
        notes="Action-expert attention score multiply over 8 heads, 51 suffix queries, and 867 keys.",
    ),
    WorkloadKernelSpec(
        name="expert_attention_context",
        stage="Action expert",
        operator="bmm",
        calls=180,
        total_flops=32_600_309_760,
        shape=(8, 51, 867, 256),
        notes="Action-expert attention context multiply over 8 heads, 51 suffix queries, and 867 keys.",
    ),
    WorkloadKernelSpec(
        name="vlm_attention_k_v",
        stage="Gemma prefix",
        operator="mm",
        calls=36,
        total_flops=30_802_968_576,
        shape=(816, 2048, 256),
        notes="Gemma prefix key/value projections 2048->256.",
    ),
    WorkloadKernelSpec(
        name="expert_attention_k_v",
        stage="Action expert",
        operator="mm",
        calls=360,
        total_flops=9_625_927_680,
        shape=(51, 1024, 256),
        notes="Action-expert key/value projections 1024->256.",
    ),
)


def _ops_per_second(value_ops_per_cycle: float, core_frequency_hz: float | None) -> float | None:
    if core_frequency_hz is None:
        return None
    return value_ops_per_cycle * core_frequency_hz


def _bound_name(limit: float, compute_peak: float) -> str:
    return "compute" if math.isclose(limit, compute_peak) else "memory"


def derive_roofline_metrics(
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> RooflineMetrics:
    """Derive normalized roofline metrics from the current model configuration."""

    dram_bw = (
        config.bandwidth.offchip_link_width_bytes
        / config.bandwidth.offchip_link_core_cycles_per_beat
    )
    vmem_bw = (
        config.bandwidth.vmem_bus_width_bytes
        / config.bandwidth.vmem_bus_core_cycles_per_beat
    )

    mxu_tile_ops = (
        2
        * config.tensor.mreg_rows
        * config.tensor.weight_tile_cols_fp8
        * config.tensor.weight_tile_rows
    )
    mxu_peak_per_mxu = mxu_tile_ops / config.matmul_latency_cycles
    mxu_peak_total = mxu_peak_per_mxu * config.tensor.mxu_count

    vpu_tile_ops = config.mreg_bytes // 2
    vpu_peak = vpu_tile_ops / config.vpu_simple_op_latency_cycles
    mxu_tile_bytes = (
        config.mreg_bytes
        + config.weight_slot_bytes
        + (config.tensor.mreg_rows * config.tensor.weight_tile_cols_fp8 * 2)
    )
    vpu_unary_tile_bytes = config.mreg_bytes * 2

    return RooflineMetrics(
        dram_bandwidth_bytes_per_cycle=dram_bw,
        vmem_bandwidth_bytes_per_cycle=vmem_bw,
        mxu_tile_ops=mxu_tile_ops,
        mxu_peak_ops_per_cycle_per_mxu=mxu_peak_per_mxu,
        mxu_peak_ops_per_cycle_total=mxu_peak_total,
        vpu_tile_ops=vpu_tile_ops,
        vpu_peak_ops_per_cycle=vpu_peak,
        mxu_dram_knee_ops_per_byte=mxu_peak_total / dram_bw,
        mxu_vmem_knee_ops_per_byte=mxu_peak_total / vmem_bw,
        vpu_dram_knee_ops_per_byte=vpu_peak / dram_bw,
        vpu_vmem_knee_ops_per_byte=vpu_peak / vmem_bw,
        mxu_tile_bytes=mxu_tile_bytes,
        vpu_unary_tile_bytes=vpu_unary_tile_bytes,
    )


def make_roofline_point(
    *,
    name: str,
    ops: int,
    bytes_moved: int,
    compute_peak_ops_per_cycle: float,
    metrics: RooflineMetrics,
) -> RooflinePoint:
    """Project one kernel onto the DRAM and VMEM roofs."""

    intensity = ops / bytes_moved if bytes_moved > 0 else math.inf
    dram_ceiling = min(compute_peak_ops_per_cycle, intensity * metrics.dram_bandwidth_bytes_per_cycle)
    vmem_ceiling = min(compute_peak_ops_per_cycle, intensity * metrics.vmem_bandwidth_bytes_per_cycle)
    return RooflinePoint(
        name=name,
        ops=ops,
        bytes_moved=bytes_moved,
        arithmetic_intensity_ops_per_byte=intensity,
        compute_peak_ops_per_cycle=compute_peak_ops_per_cycle,
        dram_ceiling_ops_per_cycle=dram_ceiling,
        vmem_ceiling_ops_per_cycle=vmem_ceiling,
        dram_bound=_bound_name(dram_ceiling, compute_peak_ops_per_cycle),
        vmem_bound=_bound_name(vmem_ceiling, compute_peak_ops_per_cycle),
    )


def representative_kernel_points(
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> list[RooflinePoint]:
    """Return representative current-design kernels for the roofline report."""

    metrics = derive_roofline_metrics(config)

    return [
        make_roofline_point(
            name="dual-mxu matmul tile",
            ops=metrics.mxu_tile_ops * config.tensor.mxu_count,
            bytes_moved=metrics.mxu_tile_bytes * config.tensor.mxu_count,
            compute_peak_ops_per_cycle=metrics.mxu_peak_ops_per_cycle_total,
            metrics=metrics,
        ),
        make_roofline_point(
            name="vpu unary tile",
            ops=metrics.vpu_tile_ops,
            bytes_moved=metrics.vpu_unary_tile_bytes,
            compute_peak_ops_per_cycle=metrics.vpu_peak_ops_per_cycle,
            metrics=metrics,
        ),
    ]


def pi0_dominant_kernel_specs() -> list[WorkloadKernelSpec]:
    """Return the dominant PI0 kernels extracted from the external FLOP report."""

    return list(PI0_DOMINANT_KERNEL_SPECS)


def _matrix_shape(spec: WorkloadKernelSpec) -> tuple[int, int, int, int]:
    if spec.operator == "bmm":
        batch, m_dim, k_dim, n_dim = spec.shape
        return batch, m_dim, k_dim, n_dim
    m_dim, k_dim, n_dim = spec.shape
    return 1, m_dim, k_dim, n_dim


def _dram_dense_bytes_per_call(spec: WorkloadKernelSpec) -> int:
    batch, m_dim, k_dim, n_dim = _matrix_shape(spec)
    dram_bytes = batch * (m_dim * k_dim + (k_dim * n_dim) + (2 * m_dim * n_dim))
    if spec.bias:
        dram_bytes += 2 * n_dim
    return dram_bytes


def _vmem_schedule_bytes_per_call(
    spec: WorkloadKernelSpec,
    config: PenguinCoreConfig,
) -> int:
    batch, m_dim, k_dim, n_dim = _matrix_shape(spec)
    m_tiles = math.ceil(m_dim / config.tensor.mreg_rows)
    k_tiles = math.ceil(k_dim / config.tensor.weight_tile_rows)
    n_tiles = math.ceil(n_dim / config.tensor.weight_tile_cols_fp8)

    mxu_stream_bytes = batch * m_tiles * n_tiles * k_tiles * (
        config.mreg_bytes + config.weight_slot_bytes
    )
    output_bytes = batch * m_tiles * n_tiles * config.mreg_bytes
    bias_bytes = batch * m_tiles * n_tiles * config.mreg_bytes if spec.bias else 0
    return mxu_stream_bytes + output_bytes + bias_bytes


def make_workload_roofline_point(
    spec: WorkloadKernelSpec,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> WorkloadRooflinePoint:
    """Project one dominant PI0 kernel onto the Penguin DRAM and VMEM roofs."""

    metrics = derive_roofline_metrics(config)
    dram_bytes_per_call = _dram_dense_bytes_per_call(spec)
    vmem_bytes_per_call = _vmem_schedule_bytes_per_call(spec, config)
    flops_per_call = spec.total_flops // spec.calls
    dram_intensity = flops_per_call / dram_bytes_per_call
    vmem_intensity = flops_per_call / vmem_bytes_per_call
    compute_peak = metrics.mxu_peak_ops_per_cycle_total
    dram_ceiling = min(compute_peak, dram_intensity * metrics.dram_bandwidth_bytes_per_cycle)
    vmem_ceiling = min(compute_peak, vmem_intensity * metrics.vmem_bandwidth_bytes_per_cycle)

    return WorkloadRooflinePoint(
        name=spec.name,
        stage=spec.stage,
        operator=spec.operator,
        calls=spec.calls,
        total_flops=spec.total_flops,
        flops_per_call=flops_per_call,
        share_total_flops=spec.total_flops / PI0_TOTAL_FLOPS,
        shape=spec.shape,
        dram_bytes_per_call=dram_bytes_per_call,
        vmem_bytes_per_call=vmem_bytes_per_call,
        dram_total_bytes=dram_bytes_per_call * spec.calls,
        vmem_total_bytes=vmem_bytes_per_call * spec.calls,
        dram_intensity_ops_per_byte=dram_intensity,
        vmem_intensity_ops_per_byte=vmem_intensity,
        compute_peak_ops_per_cycle=compute_peak,
        dram_ceiling_ops_per_cycle=dram_ceiling,
        vmem_ceiling_ops_per_cycle=vmem_ceiling,
        dram_bound=_bound_name(dram_ceiling, compute_peak),
        vmem_bound=_bound_name(vmem_ceiling, compute_peak),
        notes=spec.notes,
    )


def pi0_workload_roofline_points(
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> list[WorkloadRooflinePoint]:
    """Model the dominant PI0 kernels against the current Penguin machine."""

    return [
        make_workload_roofline_point(spec, config=config)
        for spec in PI0_DOMINANT_KERNEL_SPECS
    ]


def aggregate_workload_point(
    points: list[WorkloadRooflinePoint],
    *,
    name: str = "pi0_dominant_kernels_aggregate",
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> WorkloadRooflinePoint:
    """Aggregate a set of workload points into one weighted roofline point."""

    if not points:
        raise ValueError("points must not be empty")

    total_flops = sum(point.total_flops for point in points)
    total_dram_bytes = sum(point.dram_total_bytes for point in points)
    total_vmem_bytes = sum(point.vmem_total_bytes for point in points)
    dram_intensity = total_flops / total_dram_bytes
    vmem_intensity = total_flops / total_vmem_bytes
    metrics = derive_roofline_metrics(config)
    compute_peak = metrics.mxu_peak_ops_per_cycle_total
    dram_ceiling = min(compute_peak, dram_intensity * metrics.dram_bandwidth_bytes_per_cycle)
    vmem_ceiling = min(compute_peak, vmem_intensity * metrics.vmem_bandwidth_bytes_per_cycle)

    return WorkloadRooflinePoint(
        name=name,
        stage="Aggregate",
        operator="mm",
        calls=sum(point.calls for point in points),
        total_flops=total_flops,
        flops_per_call=0,
        share_total_flops=total_flops / PI0_TOTAL_FLOPS,
        shape=(),
        dram_bytes_per_call=0,
        vmem_bytes_per_call=0,
        dram_total_bytes=total_dram_bytes,
        vmem_total_bytes=total_vmem_bytes,
        dram_intensity_ops_per_byte=dram_intensity,
        vmem_intensity_ops_per_byte=vmem_intensity,
        compute_peak_ops_per_cycle=compute_peak,
        dram_ceiling_ops_per_cycle=dram_ceiling,
        vmem_ceiling_ops_per_cycle=vmem_ceiling,
        dram_bound=_bound_name(dram_ceiling, compute_peak),
        vmem_bound=_bound_name(vmem_ceiling, compute_peak),
        notes="Weighted aggregate over the dominant PI0 kernels from the external FLOP report.",
    )


def plot_roofline(
    output_path: str | Path,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
    points: list[RooflinePoint] | None = None,
) -> Path:
    """Draw the normalized roofline plot and save it to `output_path`."""

    metrics = derive_roofline_metrics(config)
    points = representative_kernel_points(config) if points is None else points
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [10 ** exponent for exponent in (-3, -2, -1, 0, 1, 2, 3, 4)]
    dram_roof = [
        min(metrics.mxu_peak_ops_per_cycle_total, x * metrics.dram_bandwidth_bytes_per_cycle)
        for x in x_values
    ]
    vmem_roof = [
        min(metrics.mxu_peak_ops_per_cycle_total, x * metrics.vmem_bandwidth_bytes_per_cycle)
        for x in x_values
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(x_values, dram_roof, label="DRAM roof", linewidth=2.5, color="firebrick")
    ax.loglog(x_values, vmem_roof, label="VMEM roof", linewidth=2.5, color="navy")
    ax.axhline(
        metrics.mxu_peak_ops_per_cycle_total,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="MXU peak compute",
    )
    ax.axhline(
        metrics.vpu_peak_ops_per_cycle,
        linestyle=":",
        linewidth=1.5,
        color="darkgreen",
        label="VPU simple-op peak",
    )

    markers = ["o", "s", "^", "D"]
    for index, point in enumerate(points):
        marker = markers[index % len(markers)]
        ax.scatter(
            point.arithmetic_intensity_ops_per_byte,
            point.dram_ceiling_ops_per_cycle,
            marker=marker,
            s=70,
            color="firebrick",
        )
        ax.scatter(
            point.arithmetic_intensity_ops_per_byte,
            point.vmem_ceiling_ops_per_cycle,
            marker=marker,
            s=70,
            color="navy",
            facecolors="none",
        )
        ax.annotate(
            f"{point.name}\nDRAM",
            (
                point.arithmetic_intensity_ops_per_byte,
                point.dram_ceiling_ops_per_cycle,
            ),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )
        ax.annotate(
            f"{point.name}\nVMEM",
            (
                point.arithmetic_intensity_ops_per_byte,
                point.vmem_ceiling_ops_per_cycle,
            ),
            textcoords="offset points",
            xytext=(8, -18),
            fontsize=8,
        )

    ax.set_title("Penguin Roofline Model (normalized per core cycle)")
    ax.set_xlabel("Arithmetic Intensity (ops/byte)")
    ax.set_ylabel("Throughput (ops/cycle)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_pi0_workload_roofline(
    output_path: str | Path,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
    points: list[WorkloadRooflinePoint] | None = None,
) -> Path:
    """Plot the dominant PI0 workload points against the current Penguin roofs."""

    metrics = derive_roofline_metrics(config)
    points = pi0_workload_roofline_points(config) if points is None else points
    aggregate = aggregate_workload_point(points, config=config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [10 ** exponent for exponent in (-1, 0, 1, 2, 3, 4, 5)]
    dram_roof = [
        min(metrics.mxu_peak_ops_per_cycle_total, x * metrics.dram_bandwidth_bytes_per_cycle)
        for x in x_values
    ]
    vmem_roof = [
        min(metrics.mxu_peak_ops_per_cycle_total, x * metrics.vmem_bandwidth_bytes_per_cycle)
        for x in x_values
    ]

    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.65, 1.0), wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[0, 1])

    stage_colors = {
        "Gemma prefix": "#b22222",
        "SigLIP vision": "#cc7a00",
        "Action expert": "#0b7285",
    }
    label_offsets = [
        (5, 5),
        (5, -8),
        (-12, 5),
        (-12, -8),
        (8, 10),
        (8, -14),
    ]

    ax.loglog(x_values, dram_roof, linewidth=2.5, color="firebrick", label="DRAM roof")
    ax.loglog(x_values, vmem_roof, linewidth=2.5, color="navy", label="VMEM roof")
    ax.axhline(
        metrics.mxu_peak_ops_per_cycle_total,
        linestyle="--",
        linewidth=1.4,
        color="black",
        label="MXU peak compute",
    )
    ax.set_title("DRAM and VMEM Roofs")
    ax.set_xlabel("Arithmetic Intensity (ops/byte)")
    ax.set_ylabel("Throughput (ops/cycle)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    for index, point in enumerate(points, start=1):
        color = stage_colors[point.stage]
        dx, dy = label_offsets[(index - 1) % len(label_offsets)]
        ax.plot(
            [
                point.vmem_intensity_ops_per_byte,
                point.dram_intensity_ops_per_byte,
            ],
            [
                point.vmem_ceiling_ops_per_cycle,
                point.dram_ceiling_ops_per_cycle,
            ],
            color=color,
            linewidth=0.9,
            alpha=0.55,
        )
        ax.scatter(
            point.dram_intensity_ops_per_byte,
            point.dram_ceiling_ops_per_cycle,
            s=52,
            color=color,
            alpha=0.92,
            zorder=3,
        )
        ax.scatter(
            point.vmem_intensity_ops_per_byte,
            point.vmem_ceiling_ops_per_cycle,
            s=52,
            facecolors="none",
            edgecolors=color,
            linewidths=1.4,
            zorder=3,
        )
        label_x = math.sqrt(
            point.dram_intensity_ops_per_byte * point.vmem_intensity_ops_per_byte
        )
        label_y = math.sqrt(
            point.dram_ceiling_ops_per_cycle * point.vmem_ceiling_ops_per_cycle
        )
        ax.annotate(
            str(index),
            (label_x, label_y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.8,
                "alpha": 0.9,
            },
        )

    ax.plot(
        [
            aggregate.vmem_intensity_ops_per_byte,
            aggregate.dram_intensity_ops_per_byte,
        ],
        [
            aggregate.vmem_ceiling_ops_per_cycle,
            aggregate.dram_ceiling_ops_per_cycle,
        ],
        color="#111827",
        linewidth=1.2,
        alpha=0.7,
    )
    ax.scatter(
        aggregate.dram_intensity_ops_per_byte,
        aggregate.dram_ceiling_ops_per_cycle,
        marker="*",
        s=220,
        edgecolors="firebrick",
        facecolors="firebrick",
        linewidths=1.6,
        zorder=5,
    )
    ax.scatter(
        aggregate.vmem_intensity_ops_per_byte,
        aggregate.vmem_ceiling_ops_per_cycle,
        marker="*",
        s=220,
        edgecolors="navy",
        facecolors="white",
        linewidths=1.6,
        zorder=5,
    )
    ax.annotate(
        "Agg",
        (
            math.sqrt(
                aggregate.dram_intensity_ops_per_byte
                * aggregate.vmem_intensity_ops_per_byte
            ),
            math.sqrt(
                aggregate.dram_ceiling_ops_per_cycle
                * aggregate.vmem_ceiling_ops_per_cycle
            ),
        ),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=8,
        fontweight="bold",
    )

    stage_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=color,
            label=stage,
            markersize=7,
        )
        for stage, color in stage_colors.items()
    ]
    dram_marker = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor="#111827",
        markeredgecolor="#111827",
        label="DRAM point",
        markersize=6,
    )
    vmem_marker = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor="white",
        markeredgecolor="#111827",
        label="VMEM point",
        markersize=6,
    )
    aggregate_handle = Line2D(
        [0],
        [0],
        marker="*",
        linestyle="",
        markerfacecolor="firebrick",
        markeredgecolor="firebrick",
        label="Aggregate pair",
        markersize=11,
    )
    ax.legend(
        handles=stage_handles + [dram_marker, vmem_marker, aggregate_handle],
        loc="lower right",
        fontsize=8,
        frameon=True,
    )

    ax_info.axis("off")
    ax_info.set_title("Kernel Key", loc="left")
    info_lines = [
        "Numbers on plot map to kernels below.",
        "Filled dot: DRAM point",
        "Open dot: VMEM point",
        "Star pair: weighted aggregate",
        "",
    ]
    for index, point in enumerate(points, start=1):
        info_lines.append(
            f"{index:>2}. {point.name} ({point.share_total_flops * 100:4.1f}%)"
        )
    ax_info.text(
        0.0,
        1.0,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
    )

    fig.suptitle("Penguin Roofline Model: PI0 Dominant Kernels", fontsize=14, y=0.98)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def format_report(
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
    core_frequency_hz: float | None = None,
) -> str:
    """Return a compact text report with key roofline metrics."""

    metrics = derive_roofline_metrics(config)
    points = representative_kernel_points(config)

    lines = [
        "Penguin theoretical roofline metrics",
        f"  DRAM bandwidth: {metrics.dram_bandwidth_bytes_per_cycle:.2f} B/cycle",
        f"  VMEM bandwidth: {metrics.vmem_bandwidth_bytes_per_cycle:.2f} B/cycle",
        f"  MXU tile ops: {metrics.mxu_tile_ops:,} ops",
        f"  MXU tile bytes: {metrics.mxu_tile_bytes:,} bytes per MXU",
        (
            "  MXU peak throughput:"
            f" {metrics.mxu_peak_ops_per_cycle_per_mxu:.2f} ops/cycle per MXU,"
            f" {metrics.mxu_peak_ops_per_cycle_total:.2f} ops/cycle total"
        ),
        f"  VPU unary-tile ops: {metrics.vpu_tile_ops:,} ops",
        f"  VPU unary-tile bytes: {metrics.vpu_unary_tile_bytes:,} bytes",
        f"  VPU simple-op peak throughput: {metrics.vpu_peak_ops_per_cycle:.2f} ops/cycle",
        (
            "  MXU knees:"
            f" DRAM {metrics.mxu_dram_knee_ops_per_byte:.2f} ops/byte,"
            f" VMEM {metrics.mxu_vmem_knee_ops_per_byte:.2f} ops/byte"
        ),
        (
            "  VPU knees:"
            f" DRAM {metrics.vpu_dram_knee_ops_per_byte:.2f} ops/byte,"
            f" VMEM {metrics.vpu_vmem_knee_ops_per_byte:.2f} ops/byte"
        ),
    ]

    if core_frequency_hz is not None:
        lines.extend(
            [
                f"  Core frequency: {core_frequency_hz / 1e6:.2f} MHz",
                (
                    "  DRAM bandwidth:"
                    f" {_ops_per_second(metrics.dram_bandwidth_bytes_per_cycle, core_frequency_hz) / 1e9:.6f} GB/s"
                ),
                (
                    "  VMEM bandwidth:"
                    f" {_ops_per_second(metrics.vmem_bandwidth_bytes_per_cycle, core_frequency_hz) / 1e9:.6f} GB/s"
                ),
                (
                    "  MXU total peak:"
                    f" {_ops_per_second(metrics.mxu_peak_ops_per_cycle_total, core_frequency_hz) / 1e9:.6f} GOPS"
                ),
                (
                    "  VPU simple-op peak:"
                    f" {_ops_per_second(metrics.vpu_peak_ops_per_cycle, core_frequency_hz) / 1e9:.6f} GOPS"
                ),
            ]
        )

    lines.append("Representative kernels")
    for point in points:
        lines.extend(
            [
                f"  {point.name}:",
                f"    intensity={point.arithmetic_intensity_ops_per_byte:.4f} ops/byte",
                (
                    "    DRAM ceiling="
                    f"{point.dram_ceiling_ops_per_cycle:.4f} ops/cycle"
                    f" ({point.dram_bound}-bound)"
                ),
                (
                    "    VMEM ceiling="
                    f"{point.vmem_ceiling_ops_per_cycle:.4f} ops/cycle"
                    f" ({point.vmem_bound}-bound)"
                ),
            ]
        )
    return "\n".join(lines)


def format_pi0_workload_report(
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
    core_frequency_hz: float | None = None,
    limit: int | None = None,
) -> str:
    """Return a compact PI0 workload roofline report."""

    metrics = derive_roofline_metrics(config)
    points = pi0_workload_roofline_points(config)
    if limit is not None:
        points = points[:limit]
    aggregate = aggregate_workload_point(points, config=config)

    lines = [
        "Penguin PI0 workload roofline",
        "  Workload source: Understanding-PI0 profiler report + OpenPI model structure",
        "  Prefix tokens: 816 = 3 x 256 image patches + 48 language tokens",
        "  Suffix tokens: 51 = 1 state token + 50 action tokens",
        (
            "  Dominant kernel coverage:"
            f" {aggregate.share_total_flops * 100:.2f}% of the"
            f" {PI0_TOTAL_FLOPS:,} total FLOPs"
        ),
        f"  DRAM roof bandwidth: {metrics.dram_bandwidth_bytes_per_cycle:.2f} B/cycle",
        f"  VMEM roof bandwidth: {metrics.vmem_bandwidth_bytes_per_cycle:.2f} B/cycle",
        f"  MXU peak compute: {metrics.mxu_peak_ops_per_cycle_total:.2f} ops/cycle",
        (
            "  Aggregate dominant-kernel intensity:"
            f" DRAM {aggregate.dram_intensity_ops_per_byte:.2f} ops/byte,"
            f" VMEM {aggregate.vmem_intensity_ops_per_byte:.2f} ops/byte"
        ),
        (
            "  Aggregate ceilings:"
            f" DRAM {aggregate.dram_ceiling_ops_per_cycle:.2f} ops/cycle"
            f" ({aggregate.dram_bound}-bound),"
            f" VMEM {aggregate.vmem_ceiling_ops_per_cycle:.2f} ops/cycle"
            f" ({aggregate.vmem_bound}-bound)"
        ),
    ]

    if core_frequency_hz is not None:
        lines.extend(
            [
                f"  Core frequency: {core_frequency_hz / 1e6:.2f} MHz",
                (
                    "  Aggregate ceilings:"
                    f" DRAM {_ops_per_second(aggregate.dram_ceiling_ops_per_cycle, core_frequency_hz) / 1e9:.6f} GOPS,"
                    f" VMEM {_ops_per_second(aggregate.vmem_ceiling_ops_per_cycle, core_frequency_hz) / 1e9:.6f} GOPS"
                ),
            ]
        )

    lines.append("Dominant kernels")
    for point in points:
        lines.append(
            (
                f"  {point.name}: share={point.share_total_flops * 100:5.2f}%"
                f" stage={point.stage}"
                f" DRAM_I={point.dram_intensity_ops_per_byte:7.2f}"
                f" VMEM_I={point.vmem_intensity_ops_per_byte:6.2f}"
                f" DRAM={point.dram_ceiling_ops_per_cycle:7.2f}"
                f" VMEM={point.vmem_ceiling_ops_per_cycle:7.2f}"
            )
        )

    return "\n".join(lines)


__all__ = [
    "PI0_DOMINANT_KERNEL_SPECS",
    "PI0_TOTAL_FLOPS",
    "RooflineMetrics",
    "RooflinePoint",
    "WorkloadKernelSpec",
    "WorkloadRooflinePoint",
    "aggregate_workload_point",
    "derive_roofline_metrics",
    "format_report",
    "format_pi0_workload_report",
    "make_roofline_point",
    "make_workload_roofline_point",
    "pi0_dominant_kernel_specs",
    "pi0_workload_roofline_points",
    "plot_pi0_workload_roofline",
    "plot_roofline",
    "representative_kernel_points",
]
