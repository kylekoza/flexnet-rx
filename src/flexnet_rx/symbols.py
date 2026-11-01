"""The plain-text symbol format shared by every tool, and symbol/value/bit conversions.

A frame is one line of digit pairs. Each pair is one 16-QAM symbol written as its I level
index then its Q level index, each 0 to 3 for the levels -3, -1, +1, +3. A frame that
starts ``0332`` therefore begins with the symbols (I=0, Q=3) and (I=3, Q=2).

The labelling is defined only up to the 90-degree rotation the demodulator chooses, so
symbol strings are comparable between frames demodulated by this package, not between
arbitrary decoders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IQArray = NDArray[np.complex64]
"""Raw complex64 IQ samples, as captured and saved."""

SymbolArray = IntArray
"""An ``(n, 2)`` integer array of (I index, Q index) pairs."""

LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])
"""Amplitude of each level index on the I and Q axes."""

_NIBBLE_WEIGHTS = np.array([8, 4, 2, 1])


def parse_symbols(text: str) -> SymbolArray:
    """Parse one line of digit pairs into an ``(n, 2)`` array.

    Args:
        text: Digits 0 to 3, an even number of them. Surrounding whitespace is ignored.

    Returns:
        The symbols as (I index, Q index) rows.

    Raises:
        ValueError: If the text holds anything but 0-3 or has an odd length.
    """
    digits = text.strip()
    if len(digits) % 2 or any(ch not in "0123" for ch in digits):
        raise ValueError("symbol text must be an even number of digits 0-3")
    return np.array([int(ch) for ch in digits], dtype=np.int64).reshape(-1, 2)


def format_symbols(symbols: SymbolArray) -> str:
    """Format an ``(n, 2)`` symbol array as one line of digit pairs."""
    return "".join(f"{i}{q}" for i, q in symbols)


def read_symbol_file(path: str | Path) -> SymbolArray:
    """Read a ``.sym`` file written by ``flexnet-demod``."""
    return parse_symbols(Path(path).read_text())


def symbol_values(symbols: SymbolArray) -> IntArray:
    """Map each symbol to a single value 0-15 as ``4 * I + Q``."""
    return np.asarray(symbols[:, 0] * 4 + symbols[:, 1], dtype=np.int64)


def values_to_symbols(values: IntArray) -> SymbolArray:
    """Map values 0-15 back to (I, Q) symbols; the inverse of :func:`symbol_values`."""
    return np.stack([values // 4, values % 4], axis=1)


def values_to_bits(values: IntArray) -> IntArray:
    """Expand values 0-15 into 4 bits each, most significant first."""
    return np.asarray(((values[:, None] >> np.array([3, 2, 1, 0])) & 1).ravel(), dtype=np.int64)


def bits_to_values(bits: IntArray) -> IntArray:
    """Pack bits, most significant first, 4 at a time into values 0-15."""
    return np.asarray(bits.reshape(-1, 4) @ _NIBBLE_WEIGHTS, dtype=np.int64)
