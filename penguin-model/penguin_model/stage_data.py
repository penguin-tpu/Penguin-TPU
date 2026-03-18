"""Claim-based one-entry stage buffer."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class StageData(Generic[T]):
    """Single-entry elastic buffer used for stage-to-stage handshaking."""

    def __init__(self, empty_value: T) -> None:
        self.empty_value = empty_value
        self.data: T = empty_value
        self.valid = False

    def prepare(self, data: T) -> None:
        self.data = data
        self.valid = self._is_non_empty(data)

    def claim(self) -> T:
        if self.valid:
            data = self.data
            self.data = self.empty_value
            self.valid = False
            return data
        return self.empty_value

    def peek(self) -> T:
        if self.valid:
            return self.data
        return self.empty_value

    def should_stall(self) -> bool:
        return self.valid

    def is_valid(self) -> bool:
        return self.valid

    def reset(self) -> None:
        self.data = self.empty_value
        self.valid = False

    def _is_non_empty(self, data: T) -> bool:
        if data is None:
            return False
        if isinstance(data, (list, dict)):
            return len(data) > 0
        return True
