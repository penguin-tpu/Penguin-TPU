"""Executable bundle definitions and writers for compiler outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import json5


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

    def write_json5(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(_dump_json5(self.to_json5_dict()) + "\n")
        return destination

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


@dataclass(frozen=True, slots=True)
class BundleManifest:
    """Package-level metadata for a generated executable bundle."""

    entry_symbol: str = "program"
    symbol_table: str = "program.symbols.json5"
    version: int = 1

    def to_json5_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "entry_symbol": self.entry_symbol,
            "symbol_table": self.symbol_table,
        }

    def write_json5(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(_dump_json5(self.to_json5_dict()) + "\n")
        return destination

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
    """File-level contract between compiler, model, RTL, and FPGA loaders."""

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


def write_executable_bundle(
    output_dir: str | Path,
    *,
    program_text: str,
    symbol_table: BundleSymbolTable,
    manifest: BundleManifest | None = None,
    constants: bytes = b"",
    symbol_files: Mapping[str, bytes] | None = None,
    program_name: str = "program.S",
    symbol_table_name: str | None = None,
    manifest_name: str = "manifest.json5",
    constants_name: str = "constants.bin",
) -> ExecutableBundle:
    """Write a bundle directory with checked-in assembly and a symbol-table sidecar."""

    resolved_symbol_table_name = (
        symbol_table_name
        if symbol_table_name is not None
        else _default_symbol_table_name(program_name)
    )
    if manifest is None:
        manifest = BundleManifest(symbol_table=resolved_symbol_table_name)
    elif manifest.symbol_table != resolved_symbol_table_name:
        raise ValueError(
            "Manifest symbol-table filename does not match the requested sidecar name"
        )
    if manifest.entry_symbol not in symbol_table.symbols:
        raise ValueError(f"Entry symbol '{manifest.entry_symbol}' is not defined")

    bundle = ExecutableBundle.from_directory(
        output_dir,
        program_name=program_name,
        symbol_table_name=resolved_symbol_table_name,
        manifest_name=manifest_name,
        constants_name=constants_name,
    )
    bundle.root.mkdir(parents=True, exist_ok=True)
    bundle.program.write_text(program_text)
    symbol_table.write_json5(bundle.symbol_table)
    bundle.constants.write_bytes(constants)
    manifest.write_json5(bundle.manifest)

    for relative_path, payload in (symbol_files or {}).items():
        target = bundle.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    return bundle


def _default_symbol_table_name(program_name: str) -> str:
    return f"{Path(program_name).stem}.symbols.json5"


def _dump_json5(value: object, *, indent: int = 2, level: int = 0, key: str | None = None) -> str:
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        prefix = " " * (indent * level)
        child_prefix = " " * (indent * (level + 1))
        items = []
        for child_key, child_value in value.items():
            rendered = _dump_json5(
                child_value,
                indent=indent,
                level=level + 1,
                key=str(child_key),
            )
            # Emit bare JSON5 identifier-style keys (no quotes) to match the
            # checked-in sidecar style, assuming all keys are simple identifiers.
            items.append(f"{child_prefix}{child_key}: {rendered},")
        return "{\n" + "\n".join(items) + f"\n{prefix}" + "}"

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return "[]"
        prefix = " " * (indent * level)
        child_prefix = " " * (indent * (level + 1))
        items = [
            f"{child_prefix}{_dump_json5(item, indent=indent, level=level + 1)},"
            for item in value
        ]
        return "[\n" + "\n".join(items) + f"\n{prefix}" + "]"

    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return _format_json5_int(value, key=key)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"Unsupported JSON5 value type: {type(value).__name__}")


def _format_json5_int(value: int, *, key: str | None) -> str:
    if key == "address" and value >= 0:
        width = max(8, ((value.bit_length() + 3) // 4) or 1)
        return f"0x{value:0{width}x}"
    return str(value)


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
