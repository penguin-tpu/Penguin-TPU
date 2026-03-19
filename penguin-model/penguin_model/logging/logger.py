"""Perfetto-style JSON trace logger for Penguin execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class RetireType(str, Enum):
    """Instruction retirement types recorded in the trace."""

    RETIRE = "retire"
    STOP = "stop"


@dataclass(slots=True)
class TraceLoggerConfig:
    """Output configuration for JSON trace dumping."""

    filename: str = "trace.json"
    ticks_per_cycle: int = 3
    normalize_pc_to_base: bool = True
    pc_base_address: int = 0


class TraceLogger:
    """
    Perfetto (Chrome Trace Event) logger for Penguin.

    The structure intentionally follows the referenced `npu_model.logging`
    design: one process for execution lanes, one for architectural state, and a
    JSON array of Chrome Trace Event records.
    """

    CORE_PID = 0
    ARCH_PID = 1
    MEM_PID = 2
    MEM_TID = 0

    def __init__(
        self,
        config: TraceLoggerConfig,
        process_name: str = "Penguin Scalar Core",
        lane_names: dict[int, str] | None = None,
    ) -> None:
        self.config = config
        trace_path = Path(config.filename)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(trace_path, "w", encoding="utf-8")
        self.events: list[dict[str, Any]] = []
        self.insn_labels: dict[int, str] = {}
        self.active: dict[tuple[int, str, int], int] = {}
        self.max_timestamp = 0
        self.pc_base_address = int(config.pc_base_address)
        self.lane_names = lane_names or {
            0: "IFU",
            1: "IDU",
            2: "EXU.SALU",
            3: "EXU.DMA",
            4: "EXU.TMEM",
            5: "EXU.MXU0",
            6: "EXU.MXU1",
            7: "EXU.VPU",
            8: "EXU.XLU",
            30: "DMA.XFER.CH0",
            31: "DMA.XFER.CH1",
            32: "DMA.XFER.CH2",
            33: "DMA.XFER.CH3",
            34: "DMA.XFER.CH4",
            35: "DMA.XFER.CH5",
            36: "DMA.XFER.CH6",
            37: "DMA.XFER.CH7",
        }
        self.arch_threads: dict[tuple[str, int], tuple[int, str]] = {}
        self.ts = 0

        self._write_event(
            {
                "name": "process_name",
                "ph": "M",
                "pid": TraceLogger.CORE_PID,
                "tid": 0,
                "args": {"name": process_name},
            }
        )
        for lane in sorted(self.lane_names):
            self._write_event(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": TraceLogger.CORE_PID,
                    "tid": lane,
                    "args": {"name": self.lane_names[lane]},
                }
            )
        self._write_event(
            {
                "name": "process_name",
                "ph": "M",
                "pid": TraceLogger.ARCH_PID,
                "tid": 0,
                "args": {"name": "ArchState"},
            }
        )
        self._write_event(
            {
                "name": "process_name",
                "ph": "M",
                "pid": TraceLogger.MEM_PID,
                "tid": 0,
                "args": {"name": "Memory"},
            }
        )
        self._write_event(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": TraceLogger.MEM_PID,
                "tid": TraceLogger.MEM_TID,
                "args": {"name": "memory"},
            }
        )

    def __enter__(self) -> TraceLogger:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the trace file."""
        self._emit_free_running_cycle_counter()
        ordered_events = self._ordered_events()
        self.file.write(json.dumps(ordered_events, separators=(",", ":")))
        self.file.write("\n")
        self.file.close()

    def _write_event(self, event: dict[str, Any]) -> None:
        timestamp = event.get("ts")
        if isinstance(timestamp, int):
            duration = int(event.get("dur", 0))
            self.max_timestamp = max(self.max_timestamp, timestamp + max(0, duration))
        self.events.append(event)

    def _timestamp(self, cycle: int | None) -> int:
        return self.ts if cycle is None else cycle

    def log_cycle(self, elapsed: int) -> None:
        self.ts += elapsed

    def set_pc_base_address(self, base_address: int) -> None:
        self.pc_base_address = int(base_address)

    def log_insn(self, insn_id: int, label: str) -> None:
        self.insn_labels[insn_id] = f"{insn_id}: {label}"

    def log_retire(
        self,
        insn_id: int,
        retire_type: RetireType = RetireType.RETIRE,
        *,
        lane: int = 0,
        cycle: int | None = None,
    ) -> None:
        label = self.insn_labels.get(insn_id, f"insn-{insn_id}")
        self._write_event(
            {
                "name": label,
                "cat": "retire",
                "ph": "i",
                "s": "t",
                "pid": TraceLogger.CORE_PID,
                "tid": lane,
                "ts": self._timestamp(cycle),
                "args": {"insn_id": insn_id, "retire_type": retire_type.value},
            }
        )

    def log_stage_start(
        self,
        insn_id: int,
        stage: str,
        *,
        lane: int = 0,
        cycle: int | None = None,
    ) -> None:
        key = (insn_id, stage, lane)
        if key not in self.active:
            self.active[key] = self._timestamp(cycle)

    def log_stage_end(
        self,
        insn_id: int,
        stage: str,
        *,
        lane: int = 0,
        cycle: int | None = None,
    ) -> None:
        key = (insn_id, stage, lane)
        if key not in self.active:
            return
        start_ts = self.active.pop(key)
        end_ts = self._timestamp(cycle)
        label = self.insn_labels.get(insn_id, f"insn-{insn_id}")
        self._write_event(
            {
                "name": label,
                "cat": stage,
                "ph": "X",
                "pid": TraceLogger.CORE_PID,
                "tid": lane,
                "ts": start_ts,
                "dur": max(0, end_ts - start_ts),
                "args": {"insn_id": insn_id, "stage": stage},
            }
        )

    def log_stop(self, reason: str, *, cycle: int | None = None) -> None:
        self._write_event(
            {
                "name": "stop",
                "cat": "control",
                "ph": "i",
                "s": "g",
                "pid": TraceLogger.CORE_PID,
                "tid": 0,
                "ts": self._timestamp(cycle),
                "args": {"reason": reason},
            }
        )

    def _ensure_arch_thread(self, regfile: str, index: int) -> tuple[int, str]:
        key = (regfile, index)
        if key not in self.arch_threads:
            if regfile == "xrf":
                tid = index
                name = f"{regfile}[{index:02d}]"
            elif regfile == "pc":
                tid = 1000
                name = "pc"
            elif regfile == "cycle":
                tid = 1001
                name = "cycle"
            else:
                tid = 2000 + index
                name = f"{regfile}[{index:02d}]"
            self.arch_threads[key] = (tid, name)
            self._write_event(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": TraceLogger.ARCH_PID,
                    "tid": tid,
                    "args": {"name": name},
                }
            )
        return self.arch_threads[key]

    def log_arch_value(
        self,
        regfile: str,
        index: int,
        value: int,
        *,
        cycle: int | None = None,
    ) -> None:
        if regfile == "pc" and self.config.normalize_pc_to_base:
            value = int(value) - self.pc_base_address
        tid, name = self._ensure_arch_thread(regfile, index)
        self._write_event(
            {
                "name": name,
                "ph": "C",
                "pid": TraceLogger.ARCH_PID,
                "tid": tid,
                "ts": self._timestamp(cycle),
                "args": {"value": value},
            }
        )

    def _emit_free_running_cycle_counter(self) -> None:
        tid, name = self._ensure_arch_thread("cycle", 0)
        ticks_per_cycle = max(1, self.config.ticks_per_cycle)
        final_cycle = (self.max_timestamp + ticks_per_cycle - 1) // ticks_per_cycle
        for cycle in range(final_cycle + 1):
            self.events.append(
                {
                    "name": name,
                    "ph": "C",
                    "pid": TraceLogger.ARCH_PID,
                    "tid": tid,
                    "ts": cycle * ticks_per_cycle,
                    "args": {"value": cycle},
                }
            )

    def _ordered_events(self) -> list[dict[str, Any]]:
        indexed_events = list(enumerate(self.events))
        indexed_events.sort(
            key=lambda item: (
                0 if "ts" not in item[1] else 1,
                int(item[1].get("ts", 0)),
                item[0],
            )
        )
        return [event for _, event in indexed_events]

    def log_memory_access(
        self,
        region: str,
        access_type: str,
        address: int,
        value: int,
        *,
        size: int,
        cycle: int | None = None,
    ) -> None:
        event_name = access_type
        if region == "vmem" or access_type in {"dma-read", "dma-write"}:
            event_name = f"{access_type} 0x{address:X} {size}B"
        self._write_event(
            {
                "name": event_name,
                "cat": "memory",
                "ph": "i",
                "s": "t",
                "pid": TraceLogger.MEM_PID,
                "tid": TraceLogger.MEM_TID,
                "ts": self._timestamp(cycle),
                "args": {
                    "access_type": access_type,
                    "region": region,
                    "address": address,
                    "size": size,
                    "transfer_address": address,
                    "transfer_size_bytes": size,
                    "value": value & 0xFFFF_FFFF,
                },
            }
        )
