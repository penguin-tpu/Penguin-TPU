"""Backward-compatible aliases for bandwidth-related config fragments."""

from .core_config import BandwidthConfig, DEFAULT_PENGUIN_CORE_CONFIG

BandwidthParameters = BandwidthConfig
DEFAULT_BANDWIDTH_PARAMETERS = DEFAULT_PENGUIN_CORE_CONFIG.bandwidth

__all__ = ["BandwidthParameters", "DEFAULT_BANDWIDTH_PARAMETERS"]
