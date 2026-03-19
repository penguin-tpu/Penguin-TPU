"""Staged model-package metadata and execution helpers for compiler outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import json5
import torch

from penguin_model import (
    DEFAULT_PENGUIN_CORE_CONFIG,
    ExecutableBundle,
    PenguinCoreConfig,
    Sim,
    StopReason,
    TraceLogger,
    TraceLoggerConfig,
    load_executable_bundle,
    preload_loaded_bundle_symbols,
)

from .pack import (
    ATTENTION_KEYS,
    BF16_DTYPE,
    pack_activation_matrix,
    pack_bf16_matrix,
    pack_weight_matrix,
    pad_attention_probabilities,
    pad_attention_values,
    read_bf16_matrix_symbol,
    read_transposed_bf16_matrix_symbol,
    scaled_attention_scores,
)


@dataclass(frozen=True, slots=True)
class TensorArtifact:
    """Raw tensor payload stored at the package root."""

    file: str
    dtype: str
    shape: tuple[int, ...]

    def to_json5_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_json5_dict(cls, payload: Mapping[str, object]) -> TensorArtifact:
        shape = payload.get("shape")
        if not isinstance(shape, list):
            raise ValueError("Tensor artifact field 'shape' must be a JSON array")
        return cls(
            file=_require_string(payload, "file"),
            dtype=_require_string(payload, "dtype"),
            shape=tuple(_require_int(value) for value in shape),
        )


@dataclass(frozen=True, slots=True)
class StageInputBinding:
    """Dynamic tensor input binding for one stage bundle."""

    symbol: str
    source: str
    pack_format: str
    transform: str = "identity"

    def to_json5_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "pack_format": self.pack_format,
            "transform": self.transform,
        }

    @classmethod
    def from_json5_dict(cls, payload: Mapping[str, object]) -> StageInputBinding:
        return cls(
            symbol=_require_string(payload, "symbol"),
            source=_require_string(payload, "source"),
            pack_format=_require_string(payload, "pack_format"),
            transform=_require_string(payload, "transform"),
        )


@dataclass(frozen=True, slots=True)
class StageBundleSpec:
    """One staged executable bundle plus its runtime dataflow bindings."""

    name: str
    bundle_dir: str
    program_name: str
    symbol_table_name: str
    inputs: tuple[StageInputBinding, ...]
    output_symbol: str = "output"
    output_format: str = "bf16_matrix"

    def to_json5_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bundle_dir": self.bundle_dir,
            "program_name": self.program_name,
            "symbol_table_name": self.symbol_table_name,
            "inputs": [binding.to_json5_dict() for binding in self.inputs],
            "output_symbol": self.output_symbol,
            "output_format": self.output_format,
        }

    @classmethod
    def from_json5_dict(cls, payload: Mapping[str, object]) -> StageBundleSpec:
        raw_inputs = payload.get("inputs")
        if not isinstance(raw_inputs, list):
            raise ValueError("Stage field 'inputs' must be a JSON array")
        return cls(
            name=_require_string(payload, "name"),
            bundle_dir=_require_string(payload, "bundle_dir"),
            program_name=_require_string(payload, "program_name"),
            symbol_table_name=_require_string(payload, "symbol_table_name"),
            inputs=tuple(
                StageInputBinding.from_json5_dict(binding)
                for binding in raw_inputs
                if isinstance(binding, Mapping)
            ),
            output_symbol=_require_string(payload, "output_symbol"),
            output_format=_require_string(payload, "output_format"),
        )


@dataclass(frozen=True, slots=True)
class CompiledModelManifest:
    """Package-level manifest for a staged model export."""

    model_kind: str
    input_artifact: TensorArtifact
    expected_output_artifact: TensorArtifact
    stages: tuple[StageBundleSpec, ...]
    output_stage: str
    version: int = 1

    def to_json5_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "model_kind": self.model_kind,
            "input_artifact": self.input_artifact.to_json5_dict(),
            "expected_output_artifact": self.expected_output_artifact.to_json5_dict(),
            "stages": [stage.to_json5_dict() for stage in self.stages],
            "output_stage": self.output_stage,
        }

    def write_json5(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(json.dumps(self.to_json5_dict(), indent=2) + "\n")
        return destination

    @classmethod
    def from_json5_dict(cls, payload: Mapping[str, object]) -> CompiledModelManifest:
        raw_input_artifact = payload.get("input_artifact")
        raw_expected_artifact = payload.get("expected_output_artifact")
        raw_stages = payload.get("stages")
        if not isinstance(raw_input_artifact, Mapping):
            raise ValueError("Package field 'input_artifact' must be a JSON object")
        if not isinstance(raw_expected_artifact, Mapping):
            raise ValueError("Package field 'expected_output_artifact' must be a JSON object")
        if not isinstance(raw_stages, list):
            raise ValueError("Package field 'stages' must be a JSON array")
        return cls(
            version=_require_field_int(payload, "version"),
            model_kind=_require_string(payload, "model_kind"),
            input_artifact=TensorArtifact.from_json5_dict(raw_input_artifact),
            expected_output_artifact=TensorArtifact.from_json5_dict(raw_expected_artifact),
            stages=tuple(
                StageBundleSpec.from_json5_dict(stage)
                for stage in raw_stages
                if isinstance(stage, Mapping)
            ),
            output_stage=_require_string(payload, "output_stage"),
        )

    @classmethod
    def read_json5(cls, path: str | Path) -> CompiledModelManifest:
        payload = json5.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("Compiled model manifest must decode to a JSON object")
        return cls.from_json5_dict(payload)


@dataclass(frozen=True, slots=True)
class CompiledModelPackage:
    """On-disk staged model package emitted by the compiler."""

    root: Path
    manifest: CompiledModelManifest
    manifest_path: Path

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        manifest_name: str = "model_package.json5",
    ) -> CompiledModelPackage:
        root = Path(directory)
        manifest_path = root / manifest_name
        return cls(
            root=root,
            manifest=CompiledModelManifest.read_json5(manifest_path),
            manifest_path=manifest_path,
        )

    @property
    def stage_bundles(self) -> Mapping[str, Path]:
        return {
            stage.name: self.root / stage.bundle_dir
            for stage in self.manifest.stages
        }


@dataclass(frozen=True, slots=True)
class CompiledModelRunResult:
    """Execution result for one staged model package."""

    package: CompiledModelPackage
    trace_path: Path
    output: torch.Tensor
    expected_output: torch.Tensor
    perf: object
    stage_bundles: Mapping[str, Path]


def write_tensor_artifact(path: str | Path, tensor: torch.Tensor, *, dtype: torch.dtype) -> Path:
    """Write one raw tensor artifact at the package root."""

    destination = Path(path)
    payload = tensor.to(dtype).contiguous().view(torch.uint8).reshape(-1)
    destination.write_bytes(bytes(payload.tolist()))
    return destination


def read_tensor_artifact(root: str | Path, artifact: TensorArtifact) -> torch.Tensor:
    """Read one raw tensor artifact from the package root."""

    payload = Path(root, artifact.file).read_bytes()
    raw = torch.tensor(list(payload), dtype=torch.uint8)
    dtype = _artifact_dtype(artifact.dtype)
    element_size = torch.empty((), dtype=dtype).element_size()
    expected_bytes = _numel(artifact.shape) * element_size
    if len(payload) != expected_bytes:
        raise ValueError(
            f"Tensor artifact '{artifact.file}' has {len(payload)} bytes, expected {expected_bytes}"
        )
    return raw.view(dtype).reshape(artifact.shape).clone()


def execute_compiled_model_package(
    package: CompiledModelPackage | str | Path,
    *,
    trace_path: str | Path,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> CompiledModelRunResult:
    """Execute one staged model package through `penguin-model`."""

    loaded_package = (
        package if isinstance(package, CompiledModelPackage) else CompiledModelPackage.from_directory(package)
    )
    model_input = read_tensor_artifact(loaded_package.root, loaded_package.manifest.input_artifact)
    expected_output = read_tensor_artifact(
        loaded_package.root,
        loaded_package.manifest.expected_output_artifact,
    )

    core = Sim(config=config)
    resolved_trace_path = Path(trace_path)
    resolved_trace_path.parent.mkdir(parents=True, exist_ok=True)
    stage_outputs: dict[str, torch.Tensor] = {}

    with TraceLogger(TraceLoggerConfig(filename=str(resolved_trace_path))) as trace_logger:
        for stage in loaded_package.manifest.stages:
            bundle = load_executable_bundle(
                ExecutableBundle.from_directory(
                    loaded_package.root / stage.bundle_dir,
                    program_name=stage.program_name,
                    symbol_table_name=stage.symbol_table_name,
                )
            )
            preload_loaded_bundle_symbols(core.state, bundle)
            for binding in stage.inputs:
                value = model_input if binding.source == "model_input" else stage_outputs[binding.source]
                payload = _pack_dynamic_input(_transform_tensor(value, binding.transform), binding.pack_format)
                symbol = bundle.symbol(binding.symbol)
                if symbol.region != "vmem":
                    raise ValueError(f"Dynamic bundle input '{stage.name}:{binding.symbol}' must live in VMEM")
                core.state.vmem.write(symbol.address, torch.tensor(list(payload), dtype=torch.uint8))
            core.execute(bundle.program, trace_logger=trace_logger)
            if core.state.stop_reason != StopReason.PROGRAM_END:
                raise RuntimeError(
                    f"stage '{stage.name}' stopped with {core.state.stop_reason!r}"
                )
            stage_outputs[stage.name] = _read_stage_output(core, bundle, stage)

    output = stage_outputs[loaded_package.manifest.output_stage]
    return CompiledModelRunResult(
        package=loaded_package,
        trace_path=resolved_trace_path,
        output=output,
        expected_output=expected_output,
        perf=core.perf,
        stage_bundles=loaded_package.stage_bundles,
    )


def _pack_dynamic_input(tensor: torch.Tensor, pack_format: str) -> bytes:
    if pack_format == "activation_fp8":
        return pack_activation_matrix(tensor)
    if pack_format == "bf16_matrix":
        return pack_bf16_matrix(tensor)
    if pack_format == "weight_fp8":
        return pack_weight_matrix(tensor)
    raise ValueError(f"Unsupported dynamic input pack format '{pack_format}'")


def _transform_tensor(tensor: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "identity":
        return tensor
    if transform == "scaled_attention_scores":
        return scaled_attention_scores(tensor)
    if transform == "pad_attention_probabilities":
        return pad_attention_probabilities(tensor)
    if transform == "pad_attention_values":
        return pad_attention_values(tensor)
    if transform == "attention_score_weights":
        return tensor[:, :ATTENTION_KEYS]
    raise ValueError(f"Unsupported tensor transform '{transform}'")


def _read_stage_output(core: Sim, bundle, stage: StageBundleSpec) -> torch.Tensor:
    symbol = bundle.symbol(stage.output_symbol)
    if stage.output_format == "bf16_matrix":
        return read_bf16_matrix_symbol(core.state, symbol)
    if stage.output_format == "bf16_transposed_matrix":
        return read_transposed_bf16_matrix_symbol(core.state, symbol)
    raise ValueError(f"Unsupported stage output format '{stage.output_format}'")


def _artifact_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported tensor artifact dtype '{name}'")


def _numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _require_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Field '{key}' must be a string")
    return value


def _require_int(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("Shape entries must be integers")
    return value


def _require_field_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Field '{key}' must be an integer")
    return value
