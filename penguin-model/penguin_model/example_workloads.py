"""Runnable tensor-example workloads backed by checked-in assembly programs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from .arch_state import ArchState, PerformanceCounters, StopReason
from .bundle import load_mapped_program
from .core_config import DEFAULT_PENGUIN_CORE_CONFIG, PenguinCoreConfig
from .core import PenguinCore
from .tensor import (
    BF16_DTYPE,
    FP8_DTYPE,
    bf16_tile_from_bytes,
    bf16_tile_to_bytes,
    fp8_tile_to_bytes,
    weight_tile_to_bytes,
)

_PROGRAM_ROOT = Path(__file__).resolve().parents[2] / "tests" / "vectors" / "programs" / "tensor" / "examples"
_DEFAULT_TRACE_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "examples"

@dataclass(frozen=True, slots=True)
class _ExampleAddresses:
    activation0: int
    activation1: int
    weight0: int
    weight1: int
    bias0: int
    bias1: int
    bias2: int
    output00: int
    output01: int
    output10: int
    output11: int
    dma_act_scratch: int
    dma_weight_scratch: int
    dma_output_scratch: int
    dma_act_dram_base: int
    dma_weight_dram_base: int
    dma_output_dram_base: int


@dataclass(frozen=True, slots=True)
class ExampleRunResult:
    name: str
    trace_path: Path
    perf: PerformanceCounters
    output: torch.Tensor
    golden: torch.Tensor


def run_matmul_example(
    trace_path: str | Path | None = None,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> ExampleRunResult:
    addresses = _example_addresses(config)
    activation = _deterministic_activation(rows=config.tensor.mreg_rows, config=config)
    weights = _deterministic_weight(cols=config.tensor.weight_tile_cols_fp8, config=config)
    golden = _bf16_reference_matmul(activation, weights)

    core = PenguinCore(config=config)
    state = core.state
    state.vmem.write(addresses.activation0, fp8_tile_to_bytes(activation, config=config))
    state.vmem.write(addresses.weight0, weight_tile_to_bytes(weights, config=config))

    resolved_trace_path = _resolve_trace_path(trace_path, "matmul_trace.json")
    perf = core.dump_json_trace(load_mapped_program(_PROGRAM_ROOT / "matmul.S"), resolved_trace_path)
    _require_program_end(core)

    output = _read_bf16_tile(state, addresses.output00, config=config)
    _require_exact_match("matmul", output, golden)
    return ExampleRunResult(
        name="matmul",
        trace_path=resolved_trace_path,
        perf=perf,
        output=output,
        golden=golden,
    )


def run_linear_example(
    trace_path: str | Path | None = None,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> ExampleRunResult:
    addresses = _example_addresses(config)
    inputs = _deterministic_activation(rows=config.tensor.mreg_rows * 2, config=config)
    weight_matrix = _deterministic_weight(
        cols=config.tensor.weight_tile_cols_fp8 * 2,
        config=config,
    )
    bias = _deterministic_bias(cols=config.tensor.weight_tile_cols_fp8 * 2)
    golden = _bf16_reference_linear(inputs, weight_matrix, bias)

    core = PenguinCore(config=config)
    state = core.state
    rows = config.tensor.mreg_rows
    cols = config.tensor.weight_tile_cols_fp8
    state.vmem.write(addresses.activation0, fp8_tile_to_bytes(inputs[:rows], config=config))
    state.vmem.write(addresses.activation1, fp8_tile_to_bytes(inputs[rows:], config=config))
    state.vmem.write(
        addresses.weight0,
        weight_tile_to_bytes(weight_matrix[:, :cols], config=config),
    )
    state.vmem.write(
        addresses.weight1,
        weight_tile_to_bytes(weight_matrix[:, cols:], config=config),
    )
    _write_bias_tiles_to_vmem(
        state,
        bias,
        addresses=(addresses.bias0, addresses.bias1),
        config=config,
    )

    resolved_trace_path = _resolve_trace_path(trace_path, "linear_trace.json")
    perf = core.dump_json_trace(load_mapped_program(_PROGRAM_ROOT / "linear.S"), resolved_trace_path)
    _require_program_end(core)

    top = torch.cat(
        (
            _read_bf16_tile(state, addresses.output00, config=config),
            _read_bf16_tile(state, addresses.output01, config=config),
        ),
        dim=1,
    )
    bottom = torch.cat(
        (
            _read_bf16_tile(state, addresses.output10, config=config),
            _read_bf16_tile(state, addresses.output11, config=config),
        ),
        dim=1,
    )
    output = torch.cat((top, bottom), dim=0)
    _require_exact_match("linear", output, golden)
    return ExampleRunResult(
        name="linear",
        trace_path=resolved_trace_path,
        perf=perf,
        output=output,
        golden=golden,
    )


def run_large_matmul_example(
    trace_path: str | Path | None = None,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> ExampleRunResult:
    return _run_dma_stripmined_example(
        name="matmul_large",
        program_name="matmul_large.S",
        trace_filename="matmul_large_trace.json",
        trace_path=trace_path,
        config=config,
        m_tiles=2,
        n_tiles=2,
        k_tiles=2,
    )


def run_large_linear_example(
    trace_path: str | Path | None = None,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> ExampleRunResult:
    return _run_dma_stripmined_example(
        name="linear_large",
        program_name="linear_large.S",
        trace_filename="linear_large_trace.json",
        trace_path=trace_path,
        config=config,
        m_tiles=3,
        n_tiles=3,
        k_tiles=2,
        bias=_deterministic_bias(cols=config.tensor.weight_tile_cols_fp8 * 3),
    )


def _resolve_trace_path(trace_path: str | Path | None, default_name: str) -> Path:
    if trace_path is None:
        resolved = _DEFAULT_TRACE_ROOT / default_name
    else:
        resolved = Path(trace_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _example_addresses(config: PenguinCoreConfig) -> _ExampleAddresses:
    vmem_base = config.memory_map.vmem.base
    dram_base = config.memory_map.dram.base
    return _ExampleAddresses(
        activation0=vmem_base + 0x0000,
        activation1=vmem_base + 0x0800,
        weight0=vmem_base + 0x1000,
        weight1=vmem_base + 0x1200,
        bias0=vmem_base + 0x1800,
        bias1=vmem_base + 0x2000,
        bias2=vmem_base + 0x2800,
        output00=vmem_base + 0x3000,
        output01=vmem_base + 0x3800,
        output10=vmem_base + 0x4000,
        output11=vmem_base + 0x4800,
        dma_act_scratch=vmem_base + 0x0000,
        dma_weight_scratch=vmem_base + 0x0800,
        dma_output_scratch=vmem_base + 0x1000,
        dma_act_dram_base=dram_base + 0x0100_0000,
        dma_weight_dram_base=dram_base + 0x0200_0000,
        dma_output_dram_base=dram_base + 0x0300_0000,
    )


def _deterministic_activation(
    *,
    rows: int,
    cols: int | None = None,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    if cols is None:
        cols = config.tensor.weight_tile_rows
    indices = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return ((indices % 17) - 8) / 8


def _deterministic_weight(
    *,
    cols: int,
    rows: int | None = None,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    if rows is None:
        rows = config.tensor.weight_tile_rows
    indices = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return (((indices * 3) % 19) - 9) / 7


def _deterministic_bias(*, cols: int) -> torch.Tensor:
    indices = torch.arange(cols, dtype=torch.float32)
    return ((indices % 13) - 6) / 11


def _quantize_fp8(values: torch.Tensor) -> torch.Tensor:
    return values.to(dtype=FP8_DTYPE).to(dtype=torch.float32)


def _bf16_reference_matmul(activation: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    activation_fp8 = _quantize_fp8(activation)
    weight_fp8 = _quantize_fp8(weights)
    acc = torch.zeros((activation_fp8.shape[0], weight_fp8.shape[1]), dtype=BF16_DTYPE)
    for inner in range(weight_fp8.shape[0]):
        product = activation_fp8[:, inner].unsqueeze(1) * weight_fp8[inner, :].unsqueeze(0)
        acc = (acc.to(torch.float32) + product).to(BF16_DTYPE)
    return acc


def _bf16_reference_linear(
    activation: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    matmul = _bf16_reference_matmul(activation, weights)
    bias_bf16 = bias.to(BF16_DTYPE).reshape(1, -1).expand(matmul.shape[0], -1)
    return (matmul.to(torch.float32) + bias_bf16.to(torch.float32)).to(BF16_DTYPE)


def _read_bf16_tile(
    state,
    address: int,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    raw = state.vmem.read(address, config.mreg_bytes).clone()
    return bf16_tile_from_bytes(raw, config=config).clone()


def _read_bf16_tile_from_memory(
    memory,
    address: int,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    raw = memory.read(address, config.mreg_bytes).clone()
    return bf16_tile_from_bytes(raw, config=config).clone()


def _write_bias_tiles_to_vmem(
    state: ArchState,
    bias: torch.Tensor,
    *,
    addresses: tuple[int, ...],
    config: PenguinCoreConfig,
) -> None:
    cols = config.tensor.weight_tile_cols_fp8
    for tile_index, address in enumerate(addresses):
        start = tile_index * cols
        if start >= bias.numel():
            break
        bias_slice = bias[start : start + cols].to(BF16_DTYPE).reshape(1, cols)
        bias_tile = bias_slice.expand(config.tensor.mreg_rows, cols).clone()
        state.vmem.write(address, bf16_tile_to_bytes(bias_tile, config=config))


def _activation_tile_address(
    m_tile: int,
    k_tile: int,
    *,
    k_tiles: int,
    addresses: _ExampleAddresses,
    config: PenguinCoreConfig,
) -> int:
    return addresses.dma_act_dram_base + (m_tile * k_tiles + k_tile) * config.mreg_bytes


def _weight_tile_address(
    k_tile: int,
    n_tile: int,
    *,
    n_tiles: int,
    addresses: _ExampleAddresses,
    config: PenguinCoreConfig,
) -> int:
    return addresses.dma_weight_dram_base + (k_tile * n_tiles + n_tile) * config.weight_slot_bytes


def _output_tile_address(
    m_tile: int,
    n_tile: int,
    *,
    n_tiles: int,
    addresses: _ExampleAddresses,
    config: PenguinCoreConfig,
) -> int:
    return addresses.dma_output_dram_base + (m_tile * n_tiles + n_tile) * config.mreg_bytes


def _write_tiled_inputs_to_dram(
    state: ArchState,
    activation: torch.Tensor,
    weights: torch.Tensor,
    *,
    addresses: _ExampleAddresses,
    config: PenguinCoreConfig,
    m_tiles: int,
    n_tiles: int,
    k_tiles: int,
) -> None:
    for m_tile in range(m_tiles):
        for k_tile in range(k_tiles):
            activation_tile = activation[
                m_tile * config.tensor.mreg_rows : (m_tile + 1) * config.tensor.mreg_rows,
                k_tile
                * config.tensor.weight_tile_rows : (k_tile + 1)
                * config.tensor.weight_tile_rows,
            ]
            state.dram.write(
                _activation_tile_address(
                    m_tile,
                    k_tile,
                    k_tiles=k_tiles,
                    addresses=addresses,
                    config=config,
                ),
                fp8_tile_to_bytes(activation_tile, config=config),
            )

    for k_tile in range(k_tiles):
        for n_tile in range(n_tiles):
            weight_tile = weights[
                k_tile
                * config.tensor.weight_tile_rows : (k_tile + 1)
                * config.tensor.weight_tile_rows,
                n_tile
                * config.tensor.weight_tile_cols_fp8 : (n_tile + 1)
                * config.tensor.weight_tile_cols_fp8,
            ]
            state.dram.write(
                _weight_tile_address(
                    k_tile,
                    n_tile,
                    n_tiles=n_tiles,
                    addresses=addresses,
                    config=config,
                ),
                weight_tile_to_bytes(weight_tile, config=config),
            )


def _read_output_from_dram(
    state: ArchState,
    *,
    addresses: _ExampleAddresses,
    config: PenguinCoreConfig,
    m_tiles: int,
    n_tiles: int,
) -> torch.Tensor:
    rows = m_tiles * config.tensor.mreg_rows
    cols = n_tiles * config.tensor.weight_tile_cols_fp8
    output = torch.zeros((rows, cols), dtype=BF16_DTYPE)

    for m_tile in range(m_tiles):
        for n_tile in range(n_tiles):
            tile = _read_bf16_tile_from_memory(
                state.dram,
                _output_tile_address(
                    m_tile,
                    n_tile,
                    n_tiles=n_tiles,
                    addresses=addresses,
                    config=config,
                ),
                config=config,
            )
            output[
                m_tile * config.tensor.mreg_rows : (m_tile + 1) * config.tensor.mreg_rows,
                n_tile
                * config.tensor.weight_tile_cols_fp8 : (n_tile + 1)
                * config.tensor.weight_tile_cols_fp8,
            ] = tile

    return output


def _run_dma_stripmined_example(
    *,
    name: str,
    program_name: str,
    trace_filename: str,
    trace_path: str | Path | None,
    config: PenguinCoreConfig,
    m_tiles: int,
    n_tiles: int,
    k_tiles: int,
    bias: torch.Tensor | None = None,
) -> ExampleRunResult:
    addresses = _example_addresses(config)
    rows = m_tiles * config.tensor.mreg_rows
    inner = k_tiles * config.tensor.weight_tile_rows
    cols = n_tiles * config.tensor.weight_tile_cols_fp8

    activation = _deterministic_activation(rows=rows, cols=inner, config=config)
    weights = _deterministic_weight(rows=inner, cols=cols, config=config)
    golden = (
        _bf16_reference_matmul(activation, weights)
        if bias is None
        else _bf16_reference_linear(activation, weights, bias)
    )

    core = PenguinCore(config=config)
    state = core.state
    _write_tiled_inputs_to_dram(
        state,
        activation,
        weights,
        addresses=addresses,
        config=config,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        k_tiles=k_tiles,
    )
    if bias is not None:
        _write_bias_tiles_to_vmem(
            state,
            bias,
            addresses=(addresses.bias0, addresses.bias1, addresses.bias2),
            config=config,
        )

    resolved_trace_path = _resolve_trace_path(trace_path, trace_filename)
    perf = core.dump_json_trace(
        load_mapped_program(_PROGRAM_ROOT / program_name),
        resolved_trace_path,
    )
    _require_program_end(core)

    output = _read_output_from_dram(
        state,
        addresses=addresses,
        config=config,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
    )
    _require_exact_match(name, output, golden)
    return ExampleRunResult(
        name=name,
        trace_path=resolved_trace_path,
        perf=perf,
        output=output,
        golden=golden,
    )


def _require_program_end(core: PenguinCore) -> None:
    if core.state.stop_reason != StopReason.PROGRAM_END:
        raise RuntimeError(f"example stopped with {core.state.stop_reason!r}")


def _require_exact_match(name: str, output: torch.Tensor, golden: torch.Tensor) -> None:
    if torch.equal(output, golden):
        return
    diff = (output.to(torch.float32) - golden.to(torch.float32)).abs()
    raise RuntimeError(f"{name} output mismatch, max_abs_error={float(diff.max().item())}")


__all__ = [
    "ExampleRunResult",
    "run_large_linear_example",
    "run_large_matmul_example",
    "run_linear_example",
    "run_matmul_example",
]
