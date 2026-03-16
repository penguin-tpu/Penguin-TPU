"""Fixed-shape Gemma-inspired workload examples built from executable bundles.

These examples follow the Gemma attention/MLP/decoder sublayer structure from the
referenced OpenPI/Hugging Face implementations, narrowed to the operations that the
current Penguin ISA can express directly today:

- matmul-backed linear projections run on the MXU
- BF16 elementwise math runs on the VPU
- BF16 whole-register transpose runs on the XLU

Unsupported Gemma pieces such as RoPE and RMSNorm are still kept on the host side
between stage bundles. The golden references mirror those same staging boundaries so the
executable-package flow stays honest about what the current hardware-visible slice
actually covers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from penguin_compiler import (
    BundleManifest as CompilerBundleManifest,
    BundleSymbol as CompilerBundleSymbol,
    BundleSymbolTable as CompilerBundleSymbolTable,
    write_executable_bundle,
)
from penguin_model import ExecutableBundle, load_executable_bundle, preload_loaded_bundle_symbols
from penguin_model.arch_state import PerformanceCounters, StopReason
from penguin_model.bundle import program_symbol_table_path
from penguin_model.core import PenguinCore
from penguin_model.core_config import DEFAULT_PENGUIN_CORE_CONFIG, PenguinCoreConfig
from penguin_model.logging import TraceLogger, TraceLoggerConfig
from penguin_model.tensor import (
    BF16_DTYPE,
    FP8_DTYPE,
    bf16_tile_from_bytes,
    bf16_tile_to_bytes,
    bf16_transposed_tile_from_bytes,
    fp8_tile_to_bytes,
    weight_tile_to_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOT = REPO_ROOT / "tests" / "vectors" / "programs" / "tensor" / "examples"
DEFAULT_TRACE_ROOT = REPO_ROOT / "outputs" / "examples"
DEFAULT_BUNDLE_ROOT = DEFAULT_TRACE_ROOT / "bundles"

ROWS = DEFAULT_PENGUIN_CORE_CONFIG.tensor.mreg_rows
HIDDEN = DEFAULT_PENGUIN_CORE_CONFIG.tensor.weight_tile_rows
TILE_COLS = DEFAULT_PENGUIN_CORE_CONFIG.tensor.weight_tile_cols_fp8
ATTENTION_KEYS = TILE_COLS


@dataclass(frozen=True, slots=True)
class GemmaExampleRunResult:
    name: str
    trace_path: Path
    bundle_root: Path
    perf: PerformanceCounters
    output: torch.Tensor
    golden: torch.Tensor
    stage_bundles: Mapping[str, Path]


def run_gemma_attention_example(
    trace_path: str | Path | None = None,
    *,
    bundle_root: str | Path | None = None,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> GemmaExampleRunResult:
    hidden = _deterministic_hidden()
    weights = _attention_weights()
    golden = _gemma_attention_reference(hidden, weights)

    core = PenguinCore(config=config)
    resolved_trace_path = _resolve_trace_path(trace_path, "gemma_attention_trace.json")
    resolved_bundle_root = _resolve_bundle_root(bundle_root, "gemma_attention")
    stage_bundles: dict[str, Path] = {}

    with TraceLogger(TraceLoggerConfig(filename=str(resolved_trace_path))) as trace_logger:
        q = _run_linear_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="q_proj",
            activation=hidden,
            weights=weights["q_proj"],
            stage_bundles=stage_bundles,
        )
        k = _run_linear_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="k_proj",
            activation=hidden,
            weights=weights["k_proj"],
            stage_bundles=stage_bundles,
        )
        v = _run_linear_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="v_proj",
            activation=hidden,
            weights=weights["v_proj"],
            stage_bundles=stage_bundles,
        )
        k_t = _run_transpose_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="k_transpose",
            matrix=k,
            stage_bundles=stage_bundles,
        )
        scores = _run_attention_scores_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="scores",
            query=q,
            key_transposed=k_t,
            stage_bundles=stage_bundles,
        )
        probs = _run_softmax_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="softmax",
            scores=scores,
            stage_bundles=stage_bundles,
        )
        context = _run_attention_context_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="context",
            probabilities=probs,
            values=v,
            stage_bundles=stage_bundles,
        )
        output = _run_linear_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="o_proj",
            activation=context,
            weights=weights["o_proj"],
            stage_bundles=stage_bundles,
        )

    _require_exact_match("gemma_attention", output, golden)
    return GemmaExampleRunResult(
        name="gemma_attention",
        trace_path=resolved_trace_path,
        bundle_root=resolved_bundle_root,
        perf=core.perf,
        output=output,
        golden=golden,
        stage_bundles=stage_bundles,
    )


def run_gemma_mlp_example(
    trace_path: str | Path | None = None,
    *,
    bundle_root: str | Path | None = None,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> GemmaExampleRunResult:
    hidden = _deterministic_hidden()
    weights = _mlp_weights()
    golden = _gemma_mlp_reference(hidden, weights)

    core = PenguinCore(config=config)
    resolved_trace_path = _resolve_trace_path(trace_path, "gemma_mlp_trace.json")
    resolved_bundle_root = _resolve_bundle_root(bundle_root, "gemma_mlp")
    stage_bundles: dict[str, Path] = {}

    with TraceLogger(TraceLoggerConfig(filename=str(resolved_trace_path))) as trace_logger:
        gate = _run_linear_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="gate_proj",
            activation=hidden,
            weights=weights["gate_proj"],
            stage_bundles=stage_bundles,
        )
        up = _run_linear_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="up_proj",
            activation=hidden,
            weights=weights["up_proj"],
            stage_bundles=stage_bundles,
        )
        gated = _run_mlp_gate_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="gate_mul",
            gate=gate,
            up=up,
            stage_bundles=stage_bundles,
        )
        output = _run_linear_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="down_proj",
            activation=gated,
            weights=weights["down_proj"],
            stage_bundles=stage_bundles,
        )

    _require_exact_match("gemma_mlp", output, golden)
    return GemmaExampleRunResult(
        name="gemma_mlp",
        trace_path=resolved_trace_path,
        bundle_root=resolved_bundle_root,
        perf=core.perf,
        output=output,
        golden=golden,
        stage_bundles=stage_bundles,
    )


def run_gemma_decoder_example(
    trace_path: str | Path | None = None,
    *,
    bundle_root: str | Path | None = None,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> GemmaExampleRunResult:
    hidden = _deterministic_hidden()
    attention_weights = _attention_weights()
    mlp_weights = _mlp_weights()
    golden = _gemma_decoder_reference(hidden, attention_weights, mlp_weights)

    core = PenguinCore(config=config)
    resolved_trace_path = _resolve_trace_path(trace_path, "gemma_decoder_trace.json")
    resolved_bundle_root = _resolve_bundle_root(bundle_root, "gemma_decoder")
    stage_bundles: dict[str, Path] = {}

    with TraceLogger(TraceLoggerConfig(filename=str(resolved_trace_path))) as trace_logger:
        attention = _run_attention_pipeline(
            core,
            trace_logger,
            resolved_bundle_root,
            hidden=hidden,
            weights=attention_weights,
            stage_prefix="attention",
            stage_bundles=stage_bundles,
        )
        hidden_bf16 = hidden.to(BF16_DTYPE)
        post_attention = _run_vadd_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="residual_after_attention",
            lhs=hidden_bf16,
            rhs=attention,
            stage_bundles=stage_bundles,
        )
        mlp = _run_mlp_pipeline(
            core,
            trace_logger,
            resolved_bundle_root,
            hidden=post_attention.to(torch.float32),
            weights=mlp_weights,
            stage_prefix="mlp",
            stage_bundles=stage_bundles,
        )
        output = _run_vadd_stage(
            core,
            trace_logger,
            resolved_bundle_root,
            stage_name="residual_after_mlp",
            lhs=post_attention,
            rhs=mlp,
            stage_bundles=stage_bundles,
        )

    _require_exact_match("gemma_decoder", output, golden)
    return GemmaExampleRunResult(
        name="gemma_decoder",
        trace_path=resolved_trace_path,
        bundle_root=resolved_bundle_root,
        perf=core.perf,
        output=output,
        golden=golden,
        stage_bundles=stage_bundles,
    )


def _run_attention_pipeline(
    core: PenguinCore,
    trace_logger: TraceLogger,
    bundle_root: Path,
    *,
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    stage_prefix: str,
    stage_bundles: dict[str, Path],
) -> torch.Tensor:
    q = _run_linear_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_q_proj",
        activation=hidden,
        weights=weights["q_proj"],
        stage_bundles=stage_bundles,
    )
    k = _run_linear_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_k_proj",
        activation=hidden,
        weights=weights["k_proj"],
        stage_bundles=stage_bundles,
    )
    v = _run_linear_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_v_proj",
        activation=hidden,
        weights=weights["v_proj"],
        stage_bundles=stage_bundles,
    )
    k_t = _run_transpose_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_k_transpose",
        matrix=k,
        stage_bundles=stage_bundles,
    )
    scores = _run_attention_scores_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_scores",
        query=q,
        key_transposed=k_t,
        stage_bundles=stage_bundles,
    )
    probabilities = _run_softmax_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_softmax",
        scores=scores,
        stage_bundles=stage_bundles,
    )
    context = _run_attention_context_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_context",
        probabilities=probabilities,
        values=v,
        stage_bundles=stage_bundles,
    )
    return _run_linear_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_o_proj",
        activation=context,
        weights=weights["o_proj"],
        stage_bundles=stage_bundles,
    )


def _run_mlp_pipeline(
    core: PenguinCore,
    trace_logger: TraceLogger,
    bundle_root: Path,
    *,
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    stage_prefix: str,
    stage_bundles: dict[str, Path],
) -> torch.Tensor:
    gate = _run_linear_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_gate_proj",
        activation=hidden,
        weights=weights["gate_proj"],
        stage_bundles=stage_bundles,
    )
    up = _run_linear_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_up_proj",
        activation=hidden,
        weights=weights["up_proj"],
        stage_bundles=stage_bundles,
    )
    gated = _run_mlp_gate_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_gate_mul",
        gate=gate,
        up=up,
        stage_bundles=stage_bundles,
    )
    return _run_linear_stage(
        core,
        trace_logger,
        bundle_root,
        stage_name=f"{stage_prefix}_down_proj",
        activation=gated,
        weights=weights["down_proj"],
        stage_bundles=stage_bundles,
    )


def _run_linear_stage(
    core: PenguinCore,
    trace_logger: TraceLogger,
    bundle_root: Path,
    *,
    stage_name: str,
    activation: torch.Tensor,
    weights: torch.Tensor,
    stage_bundles: dict[str, Path],
) -> torch.Tensor:
    loaded = _prepare_stage_bundle(
        bundle_root,
        stage_name=stage_name,
        program_name="gemma_linear_64x32.S",
        inputs={
            "activation": _pack_activation_matrix(activation),
            "weights": _pack_weight_matrix(weights),
        },
    )
    stage_bundles[stage_name] = loaded.bundle.root
    _run_loaded_stage(core, trace_logger, loaded)
    return _read_matrix_symbol(core, loaded, "output", cols=2 * TILE_COLS)


def _run_mlp_gate_stage(
    core: PenguinCore,
    trace_logger: TraceLogger,
    bundle_root: Path,
    *,
    stage_name: str,
    gate: torch.Tensor,
    up: torch.Tensor,
    stage_bundles: dict[str, Path],
) -> torch.Tensor:
    loaded = _prepare_stage_bundle(
        bundle_root,
        stage_name=stage_name,
        program_name="gemma_mlp_gate_64x32.S",
        inputs={
            "gate": _pack_bf16_matrix(gate),
            "up": _pack_bf16_matrix(up),
            "constants": _pack_gelu_constants(),
        },
    )
    stage_bundles[stage_name] = loaded.bundle.root
    _run_loaded_stage(core, trace_logger, loaded)
    return _read_matrix_symbol(core, loaded, "output", cols=2 * TILE_COLS)


def _run_vadd_stage(
    core: PenguinCore,
    trace_logger: TraceLogger,
    bundle_root: Path,
    *,
    stage_name: str,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    stage_bundles: dict[str, Path],
) -> torch.Tensor:
    loaded = _prepare_stage_bundle(
        bundle_root,
        stage_name=stage_name,
        program_name="gemma_vadd_64x32.S",
        inputs={
            "lhs": _pack_bf16_matrix(lhs),
            "rhs": _pack_bf16_matrix(rhs),
        },
    )
    stage_bundles[stage_name] = loaded.bundle.root
    _run_loaded_stage(core, trace_logger, loaded)
    return _read_matrix_symbol(core, loaded, "output", cols=2 * TILE_COLS)


def _run_softmax_stage(
    core: PenguinCore,
    trace_logger: TraceLogger,
    bundle_root: Path,
    *,
    stage_name: str,
    scores: torch.Tensor,
    stage_bundles: dict[str, Path],
) -> torch.Tensor:
    loaded = _prepare_stage_bundle(
        bundle_root,
        stage_name=stage_name,
        program_name="gemma_softmax_64x16.S",
        inputs={"input": _pack_bf16_matrix(_scaled_attention_scores(scores))},
    )
    stage_bundles[stage_name] = loaded.bundle.root
    _run_loaded_stage(core, trace_logger, loaded)
    return _read_matrix_symbol(core, loaded, "output", cols=TILE_COLS)


def _run_transpose_stage(
    core: PenguinCore,
    trace_logger: TraceLogger,
    bundle_root: Path,
    *,
    stage_name: str,
    matrix: torch.Tensor,
    stage_bundles: dict[str, Path],
) -> torch.Tensor:
    loaded = _prepare_stage_bundle(
        bundle_root,
        stage_name=stage_name,
        program_name="gemma_transpose_64x32.S",
        inputs={"input": _pack_bf16_matrix(matrix)},
    )
    stage_bundles[stage_name] = loaded.bundle.root
    _run_loaded_stage(core, trace_logger, loaded)
    return _read_transposed_matrix_symbol(core, loaded, "output", rows=2 * TILE_COLS)


def _run_attention_scores_stage(
    core: PenguinCore,
    trace_logger: TraceLogger,
    bundle_root: Path,
    *,
    stage_name: str,
    query: torch.Tensor,
    key_transposed: torch.Tensor,
    stage_bundles: dict[str, Path],
) -> torch.Tensor:
    loaded = _prepare_stage_bundle(
        bundle_root,
        stage_name=stage_name,
        program_name="gemma_attention_scores_64x16.S",
        inputs={
            "activation": _pack_activation_matrix(query),
            "weights": _pack_weight_matrix(key_transposed[:, :ATTENTION_KEYS]),
        },
    )
    stage_bundles[stage_name] = loaded.bundle.root
    _run_loaded_stage(core, trace_logger, loaded)
    return _read_matrix_symbol(core, loaded, "output", cols=TILE_COLS)


def _run_attention_context_stage(
    core: PenguinCore,
    trace_logger: TraceLogger,
    bundle_root: Path,
    *,
    stage_name: str,
    probabilities: torch.Tensor,
    values: torch.Tensor,
    stage_bundles: dict[str, Path],
) -> torch.Tensor:
    loaded = _prepare_stage_bundle(
        bundle_root,
        stage_name=stage_name,
        program_name="gemma_attention_context_64x32.S",
        inputs={
            "activation": _pack_activation_matrix(_pad_attention_probabilities(probabilities)),
            "weights": _pack_weight_matrix(_pad_attention_values(values)),
        },
    )
    stage_bundles[stage_name] = loaded.bundle.root
    _run_loaded_stage(core, trace_logger, loaded)
    return _read_matrix_symbol(core, loaded, "output", cols=2 * TILE_COLS)


def _prepare_stage_bundle(
    bundle_root: Path,
    *,
    stage_name: str,
    program_name: str,
    inputs: Mapping[str, torch.Tensor],
):
    program_path = PROGRAM_ROOT / program_name
    base_table = CompilerBundleSymbolTable.read_json5(program_symbol_table_path(program_path))
    symbol_files: dict[str, bytes] = {}
    symbols: dict[str, CompilerBundleSymbol] = {}

    for name, symbol in base_table.symbols.items():
        file_name = symbol.file
        if name in inputs:
            payload = bytes(inputs[name].reshape(-1).tolist())
            if len(payload) != symbol.size_bytes:
                raise ValueError(
                    f"Bundle payload for '{program_name}:{name}' has {len(payload)} bytes, "
                    f"expected {symbol.size_bytes}"
                )
            file_name = f"{name}.bin"
            symbol_files[file_name] = payload
        elif name != "program":
            file_name = None
        symbols[name] = CompilerBundleSymbol(
            name=symbol.name,
            kind=symbol.kind,
            region=symbol.region,
            address=symbol.address,
            size_bytes=symbol.size_bytes,
            file=file_name,
            description=symbol.description,
        )

    stage_dir = bundle_root / stage_name
    symbol_table_name = program_path.with_suffix(".symbols.json5").name
    write_executable_bundle(
        stage_dir,
        program_text=program_path.read_text(),
        program_name=program_name,
        symbol_table_name=symbol_table_name,
        manifest=CompilerBundleManifest(symbol_table=symbol_table_name),
        symbol_table=CompilerBundleSymbolTable(symbols=symbols),
        symbol_files=symbol_files,
    )
    return load_executable_bundle(
        ExecutableBundle.from_directory(
            stage_dir,
            program_name=program_name,
            symbol_table_name=symbol_table_name,
        )
    )


def _run_loaded_stage(core: PenguinCore, trace_logger: TraceLogger, loaded) -> None:
    preload_loaded_bundle_symbols(core.state, loaded)
    core.execute(loaded.program, trace_logger=trace_logger)
    if core.state.stop_reason != StopReason.PROGRAM_END:
        raise RuntimeError(f"stage '{loaded.bundle.root.name}' stopped with {core.state.stop_reason!r}")


def _read_matrix_symbol(
    core: PenguinCore,
    loaded,
    symbol_name: str,
    *,
    cols: int,
) -> torch.Tensor:
    symbol = loaded.symbol(symbol_name)
    tiles = cols // TILE_COLS
    chunks = [
        bf16_tile_from_bytes(
            core.state.vmem.read(symbol.address + index * core.config.mreg_bytes, core.config.mreg_bytes).clone(),
            config=core.config,
        ).clone()
        for index in range(tiles)
    ]
    return torch.cat(chunks, dim=1)


def _read_transposed_matrix_symbol(
    core: PenguinCore,
    loaded,
    symbol_name: str,
    *,
    rows: int,
) -> torch.Tensor:
    symbol = loaded.symbol(symbol_name)
    tiles = rows // TILE_COLS
    chunks = [
        bf16_transposed_tile_from_bytes(
            core.state.vmem.read(symbol.address + index * core.config.mreg_bytes, core.config.mreg_bytes).clone(),
            config=core.config,
        ).clone()
        for index in range(tiles)
    ]
    return torch.cat(chunks, dim=0)


def _pack_activation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    cols_per_tile = DEFAULT_PENGUIN_CORE_CONFIG.tensor.mreg_row_bytes
    if matrix.shape[0] != ROWS or matrix.shape[1] % cols_per_tile != 0:
        raise ValueError(f"Activation matrix must have shape (64, N*32), got {tuple(matrix.shape)}")
    return torch.cat(
        [
            fp8_tile_to_bytes(matrix[:, start : start + cols_per_tile])
            for start in range(0, matrix.shape[1], cols_per_tile)
        ]
    )


def _pack_bf16_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[0] != ROWS or matrix.shape[1] % TILE_COLS != 0:
        raise ValueError(f"BF16 matrix must have shape (64, N*16), got {tuple(matrix.shape)}")
    return torch.cat(
        [
            bf16_tile_to_bytes(matrix[:, start : start + TILE_COLS])
            for start in range(0, matrix.shape[1], TILE_COLS)
        ]
    )


def _scaled_attention_scores(scores: torch.Tensor) -> torch.Tensor:
    return _bf16_binary_op(
        scores,
        _scalar_tile(HIDDEN**-0.5),
        torch.mul,
    )


def _pad_attention_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    if tuple(probabilities.shape) != (ROWS, ATTENTION_KEYS):
        raise ValueError(
            f"Attention probabilities must have shape ({ROWS}, {ATTENTION_KEYS}), "
            f"got {tuple(probabilities.shape)}"
        )
    first_chunk = torch.cat(
        [
            probabilities.to(torch.float32),
            torch.zeros((ROWS, TILE_COLS), dtype=torch.float32),
        ],
        dim=1,
    )
    return torch.cat(
        [
            first_chunk,
            torch.zeros((ROWS, 2 * TILE_COLS), dtype=torch.float32),
        ],
        dim=1,
    )


def _pad_attention_values(values: torch.Tensor) -> torch.Tensor:
    if tuple(values.shape) != (ROWS, 2 * TILE_COLS):
        raise ValueError(
            f"Attention values must have shape ({ROWS}, {2 * TILE_COLS}), "
            f"got {tuple(values.shape)}"
        )
    key_slice = values[:ATTENTION_KEYS, :].to(torch.float32)
    first_chunk = torch.cat(
        [
            key_slice,
            torch.zeros((HIDDEN - ATTENTION_KEYS, 2 * TILE_COLS), dtype=torch.float32),
        ],
        dim=0,
    )
    second_chunk = torch.zeros((HIDDEN, 2 * TILE_COLS), dtype=torch.float32)
    return torch.cat([first_chunk, second_chunk], dim=0)


def _pack_weight_matrix(matrix: torch.Tensor) -> torch.Tensor:
    row_tile = DEFAULT_PENGUIN_CORE_CONFIG.tensor.weight_tile_rows
    if matrix.shape[0] % row_tile != 0 or matrix.shape[1] % TILE_COLS != 0:
        raise ValueError(f"Weight matrix must have shape (N*32, M*16), got {tuple(matrix.shape)}")
    chunks = []
    for row_start in range(0, matrix.shape[0], row_tile):
        for col_start in range(0, matrix.shape[1], TILE_COLS):
            chunks.append(weight_tile_to_bytes(matrix[row_start : row_start + row_tile, col_start : col_start + TILE_COLS]))
    return torch.cat(chunks)


def _quantize_fp8(values: torch.Tensor) -> torch.Tensor:
    return values.to(FP8_DTYPE).to(torch.float32)


def _bf16_reference_matmul(activation: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    activation_fp8 = _quantize_fp8(activation)
    weight_fp8 = _quantize_fp8(weights)
    acc = torch.zeros((activation_fp8.shape[0], weight_fp8.shape[1]), dtype=BF16_DTYPE)
    for inner in range(weight_fp8.shape[0]):
        product = activation_fp8[:, inner].unsqueeze(1) * weight_fp8[inner, :].unsqueeze(0)
        acc = (acc.to(torch.float32) + product).to(BF16_DTYPE)
    return acc


def _gemma_attention_reference(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    q = _bf16_reference_matmul(hidden, weights["q_proj"])
    k = _bf16_reference_matmul(hidden, weights["k_proj"])
    v = _bf16_reference_matmul(hidden, weights["v_proj"])
    key_slice = k[:ATTENTION_KEYS, :].transpose(0, 1).contiguous().to(BF16_DTYPE)
    scores = _bf16_reference_matmul(q.to(torch.float32), key_slice.to(torch.float32))
    probabilities = _bf16_softmax_decomposition(_scaled_attention_scores(scores))
    context = _bf16_reference_matmul(probabilities.to(torch.float32), v[:ATTENTION_KEYS, :].to(torch.float32))
    return _bf16_reference_matmul(context.to(torch.float32), weights["o_proj"])


def _gemma_mlp_reference(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    gate = _bf16_reference_matmul(hidden, weights["gate_proj"])
    up = _bf16_reference_matmul(hidden, weights["up_proj"])
    activated = _bf16_gelu_decomposition(gate)
    gated = _bf16_binary_op(activated, up, torch.mul)
    return _bf16_reference_matmul(gated.to(torch.float32), weights["down_proj"])


def _bf16_round(values: torch.Tensor) -> torch.Tensor:
    return values.to(BF16_DTYPE).to(torch.float32)


def _bf16_binary_op(lhs: torch.Tensor, rhs: torch.Tensor, op) -> torch.Tensor:
    return _bf16_round(op(lhs.to(torch.float32), rhs.to(torch.float32)))


def _bf16_unary_op(src: torch.Tensor, op) -> torch.Tensor:
    return _bf16_round(op(src.to(torch.float32)))


def _bf16_row_reduce_max(src: torch.Tensor) -> torch.Tensor:
    reduced = torch.amax(src.to(torch.float32), dim=1, keepdim=True).expand_as(src)
    return _bf16_round(reduced)


def _bf16_row_reduce_sum(src: torch.Tensor) -> torch.Tensor:
    reduced = torch.sum(src.to(torch.float32), dim=1, keepdim=True).expand_as(src)
    return _bf16_round(reduced)


def _scalar_tile(value: float) -> torch.Tensor:
    return torch.full((ROWS, TILE_COLS), value, dtype=torch.float32)


def _pack_gelu_constants() -> torch.Tensor:
    return torch.cat(
        [
            bf16_tile_to_bytes(_scalar_tile(0.0)),
            bf16_tile_to_bytes(_scalar_tile(1.0)),
            bf16_tile_to_bytes(_scalar_tile(2.0)),
            bf16_tile_to_bytes(_scalar_tile(0.5)),
            bf16_tile_to_bytes(_scalar_tile(math.sqrt(2.0 / math.pi))),
            bf16_tile_to_bytes(_scalar_tile(0.044715)),
        ]
    )


def _bf16_gelu_decomposition(src: torch.Tensor) -> torch.Tensor:
    if src.shape[1] % TILE_COLS != 0:
        raise ValueError(f"GELU source width must be a multiple of {TILE_COLS}, got {tuple(src.shape)}")
    return torch.cat(
        [
            _bf16_gelu_tile(src[:, start : start + TILE_COLS])
            for start in range(0, src.shape[1], TILE_COLS)
        ],
        dim=1,
    )


def _bf16_gelu_tile(src: torch.Tensor) -> torch.Tensor:
    zero = _scalar_tile(0.0)
    one = _scalar_tile(1.0)
    two = _scalar_tile(2.0)
    half = _scalar_tile(0.5)
    alpha = _scalar_tile(math.sqrt(2.0 / math.pi))
    beta = _scalar_tile(0.044715)

    squared = _bf16_binary_op(src, src, torch.mul)
    cubed = _bf16_binary_op(squared, src, torch.mul)
    beta_cubed = _bf16_binary_op(cubed, beta, torch.mul)
    inner = _bf16_binary_op(src, beta_cubed, torch.add)
    z = _bf16_binary_op(inner, alpha, torch.mul)
    two_z = _bf16_binary_op(z, two, torch.mul)
    neg_two_z = _bf16_binary_op(zero, two_z, torch.sub)
    exp_neg = _bf16_unary_op(neg_two_z, torch.exp)
    denom = _bf16_binary_op(one, exp_neg, torch.add)
    sigma = _bf16_unary_op(denom, torch.reciprocal)
    twice_sigma = _bf16_binary_op(sigma, two, torch.mul)
    tanh_z = _bf16_binary_op(twice_sigma, one, torch.sub)
    one_plus_tanh = _bf16_binary_op(tanh_z, one, torch.add)
    half_x = _bf16_binary_op(src, half, torch.mul)
    return _bf16_binary_op(half_x, one_plus_tanh, torch.mul)


def _bf16_softmax_decomposition(src: torch.Tensor) -> torch.Tensor:
    row_max = _bf16_row_reduce_max(src)
    centered = _bf16_binary_op(src, row_max, torch.sub)
    exponentiated = _bf16_unary_op(centered, torch.exp)
    row_sum = _bf16_row_reduce_sum(exponentiated)
    reciprocal = _bf16_unary_op(row_sum, torch.reciprocal)
    return _bf16_binary_op(exponentiated, reciprocal, torch.mul)


def _gemma_decoder_reference(
    hidden: torch.Tensor,
    attention_weights: Mapping[str, torch.Tensor],
    mlp_weights: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    attention = _gemma_attention_reference(hidden, attention_weights)
    hidden_bf16 = hidden.to(BF16_DTYPE)
    post_attention = (hidden_bf16.to(torch.float32) + attention.to(torch.float32)).to(BF16_DTYPE)
    mlp = _gemma_mlp_reference(post_attention.to(torch.float32), mlp_weights)
    return (post_attention.to(torch.float32) + mlp.to(torch.float32)).to(BF16_DTYPE)


def _deterministic_hidden() -> torch.Tensor:
    indices = torch.arange(ROWS * HIDDEN, dtype=torch.float32).reshape(ROWS, HIDDEN)
    return ((indices % 11) / 10) + 0.5


def _attention_weights() -> dict[str, torch.Tensor]:
    return {
        "q_proj": _deterministic_weight(0.030, 0.018, twist=0),
        "k_proj": _deterministic_weight(0.024, 0.016, twist=3),
        "v_proj": _deterministic_weight(0.038, 0.022, twist=5),
        "o_proj": _deterministic_weight(0.034, 0.020, twist=7),
    }


def _mlp_weights() -> dict[str, torch.Tensor]:
    return {
        "gate_proj": _deterministic_weight(0.200, 0.120, twist=1),
        "up_proj": _deterministic_weight(0.110, 0.080, twist=4),
        "down_proj": _deterministic_weight(0.060, 0.032, twist=6),
    }


def _deterministic_weight(base: float, span: float, *, twist: int) -> torch.Tensor:
    indices = torch.arange(HIDDEN * HIDDEN, dtype=torch.float32).reshape(HIDDEN, HIDDEN)
    pattern = ((indices * (twist + 3) + (indices.transpose(0, 1) * (twist + 1))) % 17) / 16
    return base + span * pattern


def _resolve_trace_path(trace_path: str | Path | None, default_name: str) -> Path:
    resolved = DEFAULT_TRACE_ROOT / default_name if trace_path is None else Path(trace_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_bundle_root(bundle_root: str | Path | None, default_name: str) -> Path:
    resolved = DEFAULT_BUNDLE_ROOT / default_name if bundle_root is None else Path(bundle_root)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _require_exact_match(name: str, output: torch.Tensor, golden: torch.Tensor) -> None:
    if torch.equal(output, golden):
        return
    diff = (output.to(torch.float32) - golden.to(torch.float32)).abs()
    raise RuntimeError(f"{name} output mismatch, max_abs_error={float(diff.max().item())}")


__all__ = [
    "GemmaExampleRunResult",
    "run_gemma_attention_example",
    "run_gemma_decoder_example",
    "run_gemma_mlp_example",
]
