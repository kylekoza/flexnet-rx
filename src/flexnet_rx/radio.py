"""Radio defaults, FlexNet air-interface constants, and the capture file format.

Every tool reads these from here, so a capture, a synthetic burst and the demodulator
always agree on tuning, padding and file naming.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from flexnet_rx.symbols import IQArray

CENTER_HZ = 901.5e6
"""Default tuning: covers 900.3-902.7 MHz at the default rate, all FlexNet uplink seen."""

RATE = 2_400_000
"""Default RTL-SDR sample rate."""

PAD_MS = 20
"""IQ kept either side of a burst."""

SUBCARRIER_HZ = 2010.8
"""Normal-mode subcarrier frequency measured on a real meter; also the symbol rate."""

PREAMBLE_MS = 12.0
"""Length of the bare-subcarrier preamble before the first data symbol."""

_FREQ_IN_NAME = re.compile(r"_(\d+(?:\.\d+)?)MHz")


def burst_name(stamp: str, freq_hz: float, duration_ms: int) -> str:
    """Return the file name for a burst, e.g. ``<stamp>_901.0623MHz_63ms.cf32``."""
    return f"{stamp}_{freq_hz / 1e6:.4f}MHz_{duration_ms}ms.cf32"


def freq_from_name(path: str | Path) -> float | None:
    """Return the burst frequency in Hz encoded in a capture file name, if any."""
    match = _FREQ_IN_NAME.search(Path(path).name)
    return float(match.group(1)) * 1e6 if match else None


def save_cf32(path: str | Path, iq: IQArray) -> int:
    """Write complex64 IQ, little-endian, and return the bytes written."""
    data = np.asarray(iq, dtype="<c8")
    data.tofile(path)
    return int(data.nbytes)


def load_cf32(path: str | Path) -> IQArray:
    """Load a complex64 IQ file such as those ``flexnet-capture`` writes."""
    return np.fromfile(path, dtype="<c8")
