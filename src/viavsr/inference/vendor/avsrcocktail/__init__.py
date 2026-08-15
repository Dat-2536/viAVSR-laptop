"""Minimal AVSRCocktail model closure vendored for checkpoint loading."""

from .configuration import AVHubertAVSRConfig
from .modeling import AVHubertAVSR

__all__ = ["AVHubertAVSR", "AVHubertAVSRConfig"]
