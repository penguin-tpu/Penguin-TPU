"""Penguin compiler package."""

from .bundle import (
    BundleManifest,
    BundleSymbol,
    BundleSymbolTable,
    ExecutableBundle,
    write_executable_bundle,
)
from .codegen import export_pytorch_model_package, schedule_assembly_file, schedule_assembly_text
from .export import (
    CapturedFixedModel,
    FixedGemmaAttention,
    FixedGemmaDecoder,
    FixedGemmaMLP,
    capture_fixed_model,
    deterministic_hidden,
    make_fixed_gemma_attention,
    make_fixed_gemma_decoder,
    make_fixed_gemma_mlp,
)
from .model_package import (
    CompiledModelManifest,
    CompiledModelPackage,
    CompiledModelRunResult,
    StageBundleSpec,
    StageInputBinding,
    TensorArtifact,
    execute_compiled_model_package,
)
from .rtl import render_verilog_rom_init, write_verilog_rom_init

__all__ = [
    "BundleManifest",
    "BundleSymbol",
    "BundleSymbolTable",
    "CapturedFixedModel",
    "CompiledModelManifest",
    "CompiledModelPackage",
    "CompiledModelRunResult",
    "ExecutableBundle",
    "FixedGemmaAttention",
    "FixedGemmaDecoder",
    "FixedGemmaMLP",
    "StageBundleSpec",
    "StageInputBinding",
    "TensorArtifact",
    "capture_fixed_model",
    "deterministic_hidden",
    "execute_compiled_model_package",
    "export_pytorch_model_package",
    "make_fixed_gemma_attention",
    "make_fixed_gemma_decoder",
    "make_fixed_gemma_mlp",
    "render_verilog_rom_init",
    "schedule_assembly_file",
    "schedule_assembly_text",
    "write_executable_bundle",
    "write_verilog_rom_init",
]
