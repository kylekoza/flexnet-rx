"""Synthetic FlexNet-like bursts and coded frames, for testing without a meter.

:func:`synth_burst` builds what an RTL-SDR would record from a Normal-mode transmission:
a bare-subcarrier preamble, then one 16-QAM symbol per subcarrier cycle, frequency
modulated onto a carrier at some offset from the tuned frequency, with noise. The pulse
shape is rectangular, which is simpler than a real meter's but exercises every step of
the demodulator.

:func:`coded_frame` builds symbol sequences carrying a planted rate-1/2 convolutional code,
an interleaver and an arbitrary bit labelling, the structure the FlexNet specification
describes; :mod:`flexnet_rx.fec` should find it.

The command-line tool writes a synthetic burst named like a real capture, so the whole
pipeline can be tried end to end.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from flexnet_rx.fec import encode
from flexnet_rx.frame import (
    FRAME_SYMBOLS,
    HEADER,
    SYNC,
    SYNC_SYMBOLS,
    TRAILER,
    TRAILER_SYMBOLS,
)
from flexnet_rx.radio import (
    CENTER_HZ,
    PAD_MS,
    PREAMBLE_MS,
    RATE,
    SUBCARRIER_HZ,
    burst_name,
    save_cf32,
)
from flexnet_rx.symbols import (
    LEVELS,
    IntArray,
    IQArray,
    SymbolArray,
    bits_to_values,
    values_to_symbols,
)

DEFAULT_OFFSET_HZ = -438e3
"""Carrier offset of synthetic bursts: 901.062 MHz, where a real meter sat."""

TAIL_MS = 0.3
"""Unmodulated carrier after the last symbol, shorter than one symbol, as real bursts end."""

PREAMBLE_LEVEL = 4.3
"""Preamble amplitude in grid units, matching the ~4.1 kHz deviation seen on a real meter."""


def example_frame(rng: np.random.Generator, header: SymbolArray | None = None) -> SymbolArray:
    """Build a 100-symbol frame with the real layout and a random payload.

    Args:
        rng: Random source for the payload (and the header if none is given).
        header: Eight header symbols; random if omitted.

    Returns:
        A ``(100, 2)`` symbol array.
    """
    frame = rng.integers(0, 4, size=(FRAME_SYMBOLS, 2))
    frame[SYNC] = SYNC_SYMBOLS
    frame[TRAILER] = TRAILER_SYMBOLS
    frame[HEADER] = header if header is not None else rng.integers(0, 4, size=(len(HEADER), 2))
    return frame


def synth_burst(
    symbols: SymbolArray,
    rng: np.random.Generator,
    rate: float = RATE,
    offset_hz: float = DEFAULT_OFFSET_HZ,
    subcarrier_hz: float = SUBCARRIER_HZ,
    hz_per_unit: float = 950.0,
    preamble_ms: float = PREAMBLE_MS,
    snr_db: float = 30.0,
    pad_ms: float = PAD_MS,
) -> IQArray:
    """Synthesise the IQ an RTL-SDR would record from one Normal-mode burst.

    Args:
        symbols: The ``(n, 2)`` symbols to send after the preamble.
        rng: Random source for the noise.
        rate: Sample rate of the output.
        offset_hz: Carrier frequency minus the tuned frequency.
        subcarrier_hz: Subcarrier frequency, which is also the symbol rate.
        hz_per_unit: Frequency deviation per unit of the 16-QAM grid.
        preamble_ms: Minimum length of the bare-subcarrier preamble; it is rounded up to
            whole subcarrier cycles, so data starts just after it as on a real meter.
        snr_db: Carrier-to-noise ratio across the full sample rate.
        pad_ms: Noise-only padding before and after, as the capture tool saves.

    Returns:
        Complex64 samples.
    """
    n_pre = math.ceil(preamble_ms * 1e-3 * subcarrier_hz)
    values = np.concatenate(
        [np.full(n_pre, PREAMBLE_LEVEL + 0j), LEVELS[symbols[:, 0]] + 1j * LEVELS[symbols[:, 1]]]
    )
    t = np.arange(int(len(values) / subcarrier_hz * rate)) / rate
    index = np.minimum((t * subcarrier_hz).astype(int), len(values) - 1)
    deviation = hz_per_unit * np.real(values[index] * np.exp(2j * np.pi * subcarrier_hz * t))
    tail = np.zeros(round(TAIL_MS * 1e-3 * rate))  # carrier on briefly after the last symbol
    phase = 2 * np.pi * np.cumsum(offset_hz + np.concatenate([deviation, tail])) / rate
    pad = int(pad_ms * 1e-3 * rate)
    sig = np.concatenate([np.zeros(pad), np.exp(1j * phase), np.zeros(pad)])
    noise_power = 10 ** (-snr_db / 10)
    noise = rng.normal(size=(len(sig), 2)) @ np.array([1, 1j]) * np.sqrt(noise_power / 2)
    return np.asarray(sig + noise, dtype=np.complex64)


def coded_frame(
    info_bits: IntArray,
    polys: tuple[int, int],
    interleaver: IntArray,
    labelling: IntArray,
) -> SymbolArray:
    """Encode bits, interleave them, and map each 4 bits to a 16-QAM symbol.

    Args:
        info_bits: Information bits.
        polys: Rate-1/2 constraint-length-3 generator polynomials, e.g. ``(7, 5)``.
        interleaver: Transmit order: transmitted bit ``k`` is coded bit ``interleaver[k]``.
        labelling: A permutation of 0-15 mapping each 4-bit value to a symbol ``4*I + Q``.

    Returns:
        The ``(n, 2)`` symbols.
    """
    sent = encode(info_bits, *polys)[interleaver]
    return values_to_symbols(labelling[bits_to_values(sent)])


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: write a synthetic burst to try the pipeline."""
    parser = argparse.ArgumentParser(
        description="Write a synthetic FlexNet Normal-mode burst as a .cf32 capture file."
    )
    parser.add_argument("out_dir", type=Path, help="directory to write into")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=3, help="frames, same header, new payload")
    parser.add_argument("--snr-db", type=float, default=30.0)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    header = rng.integers(0, 4, size=(len(HEADER), 2))
    for k in range(args.count):
        iq = synth_burst(example_frame(rng, header), rng, snr_db=args.snr_db)
        path = args.out_dir / burst_name(f"synthetic{k:02d}", CENTER_HZ + DEFAULT_OFFSET_HZ, 63)
        save_cf32(path, iq)
        print(path)


if __name__ == "__main__":
    main()
