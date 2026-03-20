"""Spec-aligned staged lowering for the fixed Gemma-style compiler flow."""

from __future__ import annotations

from pathlib import Path

from ..bundle import BundleManifest, BundleSymbol, BundleSymbolTable, write_executable_bundle
from ..export import CapturedFixedModel, capture_fixed_model
from ..model_package import (
    CompiledModelManifest,
    CompiledModelPackage,
    StageBundleSpec,
    StageInputBinding,
    TensorArtifact,
    write_tensor_artifact,
)
from ..pack import pack_gelu_constants, pack_weight_matrix
from penguin_model import (
    DEFAULT_PENGUIN_CORE_CONFIG,
    PenguinCoreConfig,
    assemble_text,
    program_symbol_table_path,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
PROGRAM_ROOT = REPO_ROOT / "tests" / "vectors" / "programs" / "tensor" / "examples"
PACKAGE_MANIFEST_NAME = "model_package.json5"


def export_pytorch_model_package(
    model,
    example_input,
    output_dir: str | Path,
    *,
    strict: bool = False,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> CompiledModelPackage:
    """Export one supported fixed PyTorch model into a staged Penguin package."""

    captured = capture_fixed_model(model, example_input, strict=strict)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    stages = tuple(_emit_stage_bundles(captured, output_root, config=config))
    manifest = CompiledModelManifest(
        model_kind=captured.model_kind,
        input_artifact=TensorArtifact(
            file="model_input.bin",
            dtype="float32",
            shape=tuple(captured.example_input.shape),
        ),
        expected_output_artifact=TensorArtifact(
            file="expected_output.bin",
            dtype="bfloat16",
            shape=tuple(captured.golden_output.shape),
        ),
        stages=stages,
        output_stage=stages[-1].name,
    )
    manifest.write_json5(output_root / PACKAGE_MANIFEST_NAME)
    write_tensor_artifact(output_root / manifest.input_artifact.file, captured.example_input, dtype=example_input.dtype)
    write_tensor_artifact(
        output_root / manifest.expected_output_artifact.file,
        captured.golden_output,
        dtype=captured.golden_output.dtype,
    )
    return CompiledModelPackage.from_directory(output_root, manifest_name=PACKAGE_MANIFEST_NAME)


def _emit_stage_bundles(
    captured: CapturedFixedModel,
    output_root: Path,
    *,
    config: PenguinCoreConfig,
) -> list[StageBundleSpec]:
    if captured.model_kind == "gemma_attention":
        return _emit_attention_stages(
            captured,
            output_root,
            config=config,
            prefix="",
            source="model_input",
            final_output="o_proj",
        )
    if captured.model_kind == "gemma_mlp":
        return _emit_mlp_stages(
            captured,
            output_root,
            config=config,
            prefix="",
            source="model_input",
            final_output="down_proj",
        )
    if captured.model_kind == "gemma_decoder":
        stages = _emit_attention_stages(
            captured,
            output_root,
            config=config,
            prefix="attention_",
            source="model_input",
            final_output="attention_o_proj",
        )
        stages.append(
            _write_stage_bundle(
                output_root,
                stage_name="residual_after_attention",
                program_name="gemma_vadd_64x32.S",
                static_inputs={},
                dynamic_inputs=(
                    StageInputBinding("lhs", "model_input", "bf16_matrix"),
                    StageInputBinding("rhs", "attention_o_proj", "bf16_matrix"),
                ),
                config=config,
            )
        )
        stages.extend(
            _emit_mlp_stages(
                captured,
                output_root,
                config=config,
                prefix="mlp_",
                source="residual_after_attention",
                final_output="mlp_down_proj",
            )
        )
        stages.append(
            _write_stage_bundle(
                output_root,
                stage_name="residual_after_mlp",
                program_name="gemma_vadd_64x32.S",
                static_inputs={},
                dynamic_inputs=(
                    StageInputBinding("lhs", "residual_after_attention", "bf16_matrix"),
                    StageInputBinding("rhs", "mlp_down_proj", "bf16_matrix"),
                ),
                config=config,
            )
        )
        return stages
    raise AssertionError(f"Unhandled fixed model kind '{captured.model_kind}'")


def _emit_attention_stages(
    captured: CapturedFixedModel,
    output_root: Path,
    *,
    config: PenguinCoreConfig,
    prefix: str,
    source: str,
    final_output: str,
) -> list[StageBundleSpec]:
    return [
        _write_stage_bundle(
            output_root,
            stage_name=f"{prefix}q_proj",
            program_name="gemma_linear_64x32.S",
            static_inputs={"weights": pack_weight_matrix(captured.state_dict["q_proj"])},
            dynamic_inputs=(StageInputBinding("activation", source, "activation_fp8"),),
            config=config,
        ),
        _write_stage_bundle(
            output_root,
            stage_name=f"{prefix}k_proj",
            program_name="gemma_linear_64x32.S",
            static_inputs={"weights": pack_weight_matrix(captured.state_dict["k_proj"])},
            dynamic_inputs=(StageInputBinding("activation", source, "activation_fp8"),),
            config=config,
        ),
        _write_stage_bundle(
            output_root,
            stage_name=f"{prefix}v_proj",
            program_name="gemma_linear_64x32.S",
            static_inputs={"weights": pack_weight_matrix(captured.state_dict["v_proj"])},
            dynamic_inputs=(StageInputBinding("activation", source, "activation_fp8"),),
            config=config,
        ),
        _write_stage_bundle(
            output_root,
            stage_name=f"{prefix}k_transpose",
            program_name="gemma_transpose_64x32.S",
            static_inputs={},
            dynamic_inputs=(StageInputBinding("input", f"{prefix}k_proj", "bf16_matrix"),),
            output_format="bf16_transposed_matrix",
            config=config,
        ),
        _write_stage_bundle(
            output_root,
            stage_name=f"{prefix}scores",
            program_name="gemma_attention_scores_64x16.S",
            static_inputs={},
            dynamic_inputs=(
                StageInputBinding("activation", f"{prefix}q_proj", "activation_fp8"),
                StageInputBinding(
                    "weights",
                    f"{prefix}k_transpose",
                    "weight_fp8",
                    transform="attention_score_weights",
                ),
            ),
            config=config,
        ),
        _write_stage_bundle(
            output_root,
            stage_name=f"{prefix}softmax",
            program_name="gemma_softmax_64x16.S",
            static_inputs={},
            dynamic_inputs=(
                StageInputBinding(
                    "input",
                    f"{prefix}scores",
                    "bf16_matrix",
                    transform="scaled_attention_scores",
                ),
            ),
            config=config,
        ),
        _write_stage_bundle(
            output_root,
            stage_name=f"{prefix}context",
            program_name="gemma_attention_context_64x32.S",
            static_inputs={},
            dynamic_inputs=(
                StageInputBinding(
                    "activation",
                    f"{prefix}softmax",
                    "activation_fp8",
                    transform="pad_attention_probabilities",
                ),
                StageInputBinding(
                    "weights",
                    f"{prefix}v_proj",
                    "weight_fp8",
                    transform="pad_attention_values",
                ),
            ),
            config=config,
        ),
        _write_stage_bundle(
            output_root,
            stage_name=final_output,
            program_name="gemma_linear_64x32.S",
            static_inputs={"weights": pack_weight_matrix(captured.state_dict["o_proj"])},
            dynamic_inputs=(StageInputBinding("activation", f"{prefix}context", "activation_fp8"),),
            config=config,
        ),
    ]


def _emit_mlp_stages(
    captured: CapturedFixedModel,
    output_root: Path,
    *,
    config: PenguinCoreConfig,
    prefix: str,
    source: str,
    final_output: str,
) -> list[StageBundleSpec]:
    return [
        _write_stage_bundle(
            output_root,
            stage_name=f"{prefix}gate_proj",
            program_name="gemma_linear_64x32.S",
            static_inputs={"weights": pack_weight_matrix(captured.state_dict["gate_proj"])},
            dynamic_inputs=(StageInputBinding("activation", source, "activation_fp8"),),
            config=config,
        ),
        _write_stage_bundle(
            output_root,
            stage_name=f"{prefix}up_proj",
            program_name="gemma_linear_64x32.S",
            static_inputs={"weights": pack_weight_matrix(captured.state_dict["up_proj"])},
            dynamic_inputs=(StageInputBinding("activation", source, "activation_fp8"),),
            config=config,
        ),
        _write_stage_bundle(
            output_root,
            stage_name=f"{prefix}gate_mul",
            program_name="gemma_mlp_gate_64x32.S",
            static_inputs={"constants": pack_gelu_constants()},
            dynamic_inputs=(
                StageInputBinding("gate", f"{prefix}gate_proj", "bf16_matrix"),
                StageInputBinding("up", f"{prefix}up_proj", "bf16_matrix"),
            ),
            config=config,
        ),
        _write_stage_bundle(
            output_root,
            stage_name=final_output,
            program_name="gemma_linear_64x32.S",
            static_inputs={"weights": pack_weight_matrix(captured.state_dict["down_proj"])},
            dynamic_inputs=(StageInputBinding("activation", f"{prefix}gate_mul", "activation_fp8"),),
            config=config,
        ),
    ]


def _write_stage_bundle(
    output_root: Path,
    *,
    stage_name: str,
    program_name: str,
    static_inputs: dict[str, bytes],
    dynamic_inputs: tuple[StageInputBinding, ...],
    output_format: str = "bf16_matrix",
    config: PenguinCoreConfig,
) -> StageBundleSpec:
    program_path = PROGRAM_ROOT / program_name
    base_table = BundleSymbolTable.read_json5(program_symbol_table_path(program_path))
    if "program" not in base_table.symbols:
        raise ValueError(f"{program_name} is missing the required 'program' symbol")
    symbol_files: dict[str, bytes] = {}
    symbols: dict[str, BundleSymbol] = {}

    for name, symbol in base_table.symbols.items():
        file_name = symbol.file if name == "program" else None
        if name in static_inputs:
            payload = static_inputs[name]
            if len(payload) != symbol.size_bytes:
                raise ValueError(
                    f"Static payload for '{program_name}:{name}' has {len(payload)} bytes, expected {symbol.size_bytes}"
                )
            file_name = f"{name}.bin"
            symbol_files[file_name] = payload
        symbols[name] = BundleSymbol(
            name=symbol.name,
            kind=symbol.kind,
            region=symbol.region,
            address=symbol.address,
            size_bytes=symbol.size_bytes,
            file=file_name,
            description=symbol.description,
        )

    stage_dir = output_root / "stages" / stage_name
    symbol_table_name = program_path.with_suffix(".symbols.json5").name
    program_text = program_path.read_text()
    scheduled_program_size = len(
        assemble_text(
            program_text,
            base_address=base_table.symbols["program"].address,
            source_name=str(program_path),
        )
    ) * 4
    symbols["program"] = BundleSymbol(
        name=symbols["program"].name,
        kind=symbols["program"].kind,
        region=symbols["program"].region,
        address=symbols["program"].address,
        size_bytes=scheduled_program_size,
        file=symbols["program"].file,
        description=symbols["program"].description,
    )
    write_executable_bundle(
        stage_dir,
        program_text=program_text,
        program_name=program_name,
        symbol_table_name=symbol_table_name,
        manifest=BundleManifest(symbol_table=symbol_table_name),
        symbol_table=BundleSymbolTable(symbols=symbols),
        symbol_files=symbol_files,
    )
    return StageBundleSpec(
        name=stage_name,
        bundle_dir=str(Path("stages") / stage_name),
        program_name=program_name,
        symbol_table_name=symbol_table_name,
        inputs=dynamic_inputs,
        output_format=output_format,
    )
