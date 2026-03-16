"""Executable bundle definitions and loaders for the model runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import json5
import torch

from .assembler import AssemblyProgram, assemble_file

if TYPE_CHECKING:
    from .arch_state import ArchState


@dataclass(frozen=True, slots=True)
class BundleSymbol:
    """Named memory-mapped artifact within an executable package."""

    name: str
    kind: str
    region: str
    address: int
    size_bytes: int
    file: str | None = None
    description: str | None = None

    def to_json5_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind,
            "region": self.region,
            "address": self.address,
            "size_bytes": self.size_bytes,
        }
        if self.file is not None:
            payload["file"] = self.file
        if self.description is not None:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_json5_dict(
        cls,
        name: str,
        payload: Mapping[str, object],
    ) -> BundleSymbol:
        return cls(
            name=name,
            kind=_require_string(payload, "kind"),
            region=_require_string(payload, "region"),
            address=_require_int(payload, "address"),
            size_bytes=_require_int(payload, "size_bytes"),
            file=_optional_string(payload, "file"),
            description=_optional_string(payload, "description"),
        )


@dataclass(frozen=True, slots=True)
class BundleSymbolTable:
    """Sidecar symbol table stored next to the assembly file."""

    symbols: Mapping[str, BundleSymbol]

    def __post_init__(self) -> None:
        normalized: dict[str, BundleSymbol] = {}
        for name, symbol in self.symbols.items():
            if name != symbol.name:
                raise ValueError(
                    f"Symbol table key '{name}' does not match symbol name '{symbol.name}'"
                )
            normalized[name] = symbol
        object.__setattr__(self, "symbols", normalized)

    def to_json5_dict(self) -> dict[str, object]:
        return {
            "symbols": {
                name: symbol.to_json5_dict()
                for name, symbol in sorted(self.symbols.items())
            }
        }

    @classmethod
    def from_json5_dict(cls, payload: Mapping[str, object]) -> BundleSymbolTable:
        raw_symbols = payload.get("symbols")
        if not isinstance(raw_symbols, Mapping):
            raise ValueError("Symbol table field 'symbols' must be a JSON object")

        symbols: dict[str, BundleSymbol] = {}
        for name, symbol_payload in raw_symbols.items():
            if not isinstance(symbol_payload, Mapping):
                raise ValueError(f"Symbol table entry '{name}' must be a JSON object")
            symbols[str(name)] = BundleSymbol.from_json5_dict(str(name), symbol_payload)
        return cls(symbols=symbols)

    @classmethod
    def read_json5(cls, path: str | Path) -> BundleSymbolTable:
        payload = _load_json5(path)
        if not isinstance(payload, Mapping):
            raise ValueError("Bundle symbol table must decode to a JSON object")
        return cls.from_json5_dict(payload)

    def symbol(self, name: str) -> BundleSymbol:
        return self.symbols[name]


@dataclass(frozen=True, slots=True)
class BundleManifest:
    """Package-level metadata for a generated executable bundle."""

    entry_symbol: str = "program"
    symbol_table: str = "program.symbols.json5"
    version: int = 1

    @classmethod
    def from_json5_dict(cls, payload: Mapping[str, object]) -> BundleManifest:
        return cls(
            version=_require_int(payload, "version"),
            entry_symbol=_require_string(payload, "entry_symbol"),
            symbol_table=_require_string(payload, "symbol_table"),
        )

    @classmethod
    def read_json5(cls, path: str | Path) -> BundleManifest:
        payload = _load_json5(path)
        if not isinstance(payload, Mapping):
            raise ValueError("Bundle manifest must decode to a JSON object")
        return cls.from_json5_dict(payload)


@dataclass(frozen=True, slots=True)
class ExecutableBundle:
    """File-level contract consumed by the model and hardware flows."""

    program: Path
    symbol_table: Path
    manifest: Path
    constants: Path

    @property
    def root(self) -> Path:
        return self.manifest.parent

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        program_name: str = "program.S",
        symbol_table_name: str | None = None,
        manifest_name: str = "manifest.json5",
        constants_name: str = "constants.bin",
    ) -> ExecutableBundle:
        root = Path(directory)
        resolved_symbol_table_name = (
            symbol_table_name
            if symbol_table_name is not None
            else _default_symbol_table_name(program_name)
        )
        return cls(
            program=root / program_name,
            symbol_table=root / resolved_symbol_table_name,
            manifest=root / manifest_name,
            constants=root / constants_name,
        )


@dataclass(frozen=True, slots=True)
class LoadedExecutableBundle:
    """Executable package resolved into runtime-ready objects."""

    bundle: ExecutableBundle
    manifest: BundleManifest
    symbol_table: BundleSymbolTable
    program: AssemblyProgram
    constants: bytes
    symbol_data: Mapping[str, bytes]

    def symbol(self, name: str) -> BundleSymbol:
        return self.symbol_table.symbol(name)

    def symbol_path(self, name: str) -> Path | None:
        symbol = self.symbol(name)
        if symbol.file is None:
            return None
        return self.bundle.root / symbol.file


def program_symbol_table_path(program_path: str | Path) -> Path:
    path = Path(program_path)
    return path.with_name(f"{path.stem}.symbols.json5")


def load_program_symbol_table(program_path: str | Path) -> BundleSymbolTable:
    return BundleSymbolTable.read_json5(program_symbol_table_path(program_path))


def load_mapped_program(
    program_path: str | Path,
    *,
    entry_symbol: str = "program",
) -> AssemblyProgram:
    symbol_table = load_program_symbol_table(program_path)
    program_symbol = symbol_table.symbol(entry_symbol)
    if program_symbol.kind != "program":
        raise ValueError(
            f"Entry symbol '{entry_symbol}' for '{program_path}' must have kind 'program', "
            f"got '{program_symbol.kind}'"
        )
    program = assemble_file(program_path, base_address=program_symbol.address)
    actual_program_size = len(program) * 4
    if actual_program_size != program_symbol.size_bytes:
        raise ValueError(
            f"Program symbol size mismatch for '{program_path}': symbol table says "
            f"{program_symbol.size_bytes} bytes, assembly contains {actual_program_size} bytes"
        )
    return program


def load_executable_bundle(bundle: ExecutableBundle) -> LoadedExecutableBundle:
    """Load a bundle directory, parse the symbol table, and assemble the program."""

    manifest = BundleManifest.read_json5(bundle.manifest)
    symbol_table_path = bundle.root / manifest.symbol_table
    symbol_table = BundleSymbolTable.read_json5(symbol_table_path)
    if bundle.symbol_table != symbol_table_path:
        bundle = ExecutableBundle(
            program=bundle.program,
            symbol_table=symbol_table_path,
            manifest=bundle.manifest,
            constants=bundle.constants,
        )

    program_symbol = symbol_table.symbol(manifest.entry_symbol)
    if program_symbol.kind != "program":
        raise ValueError(
            f"Entry symbol '{manifest.entry_symbol}' must have kind 'program', "
            f"got '{program_symbol.kind}'"
        )

    program = load_mapped_program(bundle.program, entry_symbol=manifest.entry_symbol)

    constants = bundle.constants.read_bytes() if bundle.constants.exists() else b""
    symbol_data: dict[str, bytes] = {}
    for name, symbol in symbol_table.symbols.items():
        if symbol.file is None or name == manifest.entry_symbol:
            continue
        path = bundle.root / symbol.file
        if path.exists():
            payload = path.read_bytes()
            if len(payload) != symbol.size_bytes:
                raise ValueError(
                    f"Symbol payload size mismatch for '{name}': symbol table says "
                    f"{symbol.size_bytes} bytes, file contains {len(payload)} bytes"
                )
            symbol_data[name] = payload

    return LoadedExecutableBundle(
        bundle=bundle,
        manifest=manifest,
        symbol_table=symbol_table,
        program=program,
        constants=constants,
        symbol_data=symbol_data,
    )


def preload_loaded_bundle_symbols(state: ArchState, bundle: LoadedExecutableBundle) -> None:
    """Stage file-backed bundle payloads into the model's mapped memories."""

    for name, symbol in bundle.symbol_table.symbols.items():
        if name == bundle.manifest.entry_symbol:
            continue
        payload = bundle.symbol_data.get(name)
        if payload is None:
            continue
        _memory_for_region(state, symbol.region).write(
            symbol.address,
            torch.tensor(list(payload), dtype=torch.uint8),
        )


def _default_symbol_table_name(program_name: str) -> str:
    return f"{Path(program_name).stem}.symbols.json5"


def _memory_for_region(state: ArchState, region: str):
    normalized = region.lower()
    if normalized == "dram":
        return state.dram
    if normalized == "vmem":
        return state.vmem
    if normalized == "imem":
        return state.imem
    raise ValueError(f"Unsupported bundle symbol region '{region}'")


def _load_json5(path: str | Path) -> object:
    source = Path(path).read_text()
    try:
        # Use the json5 library to parse JSON5, including bare keys and other extensions.
        return json5.loads(source)
    except json5.JSON5Error as exc:  # type: ignore[attr-defined]
        raise ValueError(f"Failed to parse JSON5 file '{path}'") from exc


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Field '{key}' must be a string if present")
    return value


def _require_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Field '{key}' must be an integer")
    return value


def _require_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Field '{key}' must be a string")
    return value


__all__ = [
    "BundleManifest",
    "BundleSymbol",
    "BundleSymbolTable",
    "ExecutableBundle",
    "LoadedExecutableBundle",
    "load_mapped_program",
    "load_program_symbol_table",
    "load_executable_bundle",
    "preload_loaded_bundle_symbols",
    "program_symbol_table_path",
]
