"""The 100-symbol FlexNet Normal-mode frame layout, and per-position analysis across frames.

Measured on one meter's 63 ms frames and checked against neighbouring meters'::

    position  0-1    sync      03 32   same on every meter seen
    position  2+12k  header    k=0..7  same in every frame of one meter, differs between meters
    other     88     payload           changes completely from frame to frame
    position  98-99  trailer   21 12   same on every meter seen

Frames from weaker meters often lose their first symbol to burst detection, so frames are
aligned on their end (the trailer) rather than their start.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from flexnet_rx.symbols import (
    IntArray,
    SymbolArray,
    format_symbols,
    read_symbol_file,
    symbol_values,
)

FRAME_SYMBOLS = 100
SYNC = np.array([0, 1])
TRAILER = np.array([98, 99])
HEADER = 2 + 12 * np.arange(8)
PAYLOAD = np.setdiff1d(np.arange(FRAME_SYMBOLS), np.concatenate([SYNC, TRAILER, HEADER]))

SYNC_SYMBOLS = np.array([[0, 3], [3, 2]])
"""The sync symbols ``03 32`` seen on every meter."""

TRAILER_SYMBOLS = np.array([[2, 1], [1, 2]])
"""The trailer symbols ``21 12`` seen on every meter."""

MISSING = -1
"""Marks a symbol position the burst did not cover."""


@dataclass(frozen=True)
class Frame:
    """One frame split into its fields.

    Attributes:
        sync: The two sync symbols.
        header: The eight header symbols, in order.
        payload: The 88 payload symbols, in order.
        trailer: The two trailer symbols.
    """

    sync: SymbolArray
    header: SymbolArray
    payload: SymbolArray
    trailer: SymbolArray


TRAILER_SEARCH = 3
"""How many symbols from the end of a burst to look for the trailer."""


def frame_end(symbols: SymbolArray) -> int:
    """Return the index just past the trailer, or the burst's end if none is found.

    Bursts can carry a stray symbol or two after the frame, so the trailer is looked for
    within the last :data:`TRAILER_SEARCH` symbols, latest match first.
    """
    for end in range(len(symbols), max(len(TRAILER), len(symbols) - TRAILER_SEARCH) - 1, -1):
        if (symbols[end - len(TRAILER) : end] == TRAILER_SYMBOLS).all():
            return end
    return len(symbols)


def align_to_trailer(symbols: SymbolArray, length: int = FRAME_SYMBOLS) -> SymbolArray:
    """Right-align a frame on its trailer to ``length`` symbols, marking missing ones.

    Args:
        symbols: Demodulated symbols, possibly a symbol or two short at the start.
        length: Full frame length.

    Returns:
        A ``(length, 2)`` array whose last rows are the trailer (see :func:`frame_end`).
        Positions the burst did not cover hold :data:`MISSING`; extra leading symbols are
        dropped.
    """
    symbols = symbols[: frame_end(symbols)]
    out = np.full((length, 2), MISSING, dtype=np.int64)
    take = min(length, len(symbols))
    out[length - take :] = symbols[len(symbols) - take :]
    return out


def split_frame(symbols: SymbolArray) -> Frame:
    """Split a frame into sync, header, payload and trailer after aligning it."""
    aligned = align_to_trailer(symbols)
    return Frame(
        sync=aligned[SYNC],
        header=aligned[HEADER],
        payload=aligned[PAYLOAD],
        trailer=aligned[TRAILER],
    )


def payload_values(symbols: SymbolArray, whole_frame: bool = False) -> IntArray:
    """Return a frame's payload (or every symbol) as values 0-15, without missing ones.

    Frames are aligned on the trailer, so the only positions a short burst can miss are
    leading ones. Dropping them keeps every remaining value in its order.

    Args:
        symbols: Demodulated symbols of one frame.
        whole_frame: Return every symbol rather than only the payload.

    Returns:
        Symbol values ``4 * I + Q`` for the covered positions.
    """
    chosen = symbols if whole_frame else split_frame(symbols).payload
    covered = (chosen != MISSING).all(axis=1)
    return symbol_values(chosen[covered])


def load_values(paths: list[Path], whole_frame: bool = False) -> list[IntArray]:
    """Read ``.sym`` files and return each frame's :func:`payload_values`."""
    return [payload_values(read_symbol_file(p), whole_frame) for p in paths]


def constant_positions(frames: list[SymbolArray]) -> NDArray[np.bool_]:
    """Return which positions hold the same symbol in every frame.

    Args:
        frames: Aligned ``(n, 2)`` frames of equal length.

    Returns:
        A boolean per position. Positions missing from any frame count as not constant.
    """
    stack = np.stack(frames)
    same = (stack == stack[0]).all(axis=(0, 2))
    covered = (stack != MISSING).all(axis=(0, 2))
    return np.asarray(same & covered)


def symbol_distance(a: SymbolArray, b: SymbolArray) -> int:
    """Count positions where two aligned frames differ, ignoring missing positions."""
    covered = (a != MISSING).all(axis=1) & (b != MISSING).all(axis=1)
    return int(((a != b).any(axis=1) & covered).sum())


def report(frames: list[SymbolArray], names: list[str]) -> list[str]:
    """Describe the constant positions, headers and pairwise distances of some frames.

    Args:
        frames: Demodulated frames, ideally all from one meter.
        names: A label for each frame.

    Returns:
        Report lines.
    """
    aligned = [align_to_trailer(f) for f in frames]
    const = constant_positions(aligned)
    layout = np.full(FRAME_SYMBOLS, ".")
    layout[np.concatenate([SYNC, TRAILER])] = "s"
    layout[HEADER] = "h"
    lines = [
        f"{len(frames)} frames aligned on the trailer to {FRAME_SYMBOLS} symbols",
        "layout   : " + "".join(layout),
        "constant : " + "".join("#" if c else "." for c in const),
        f"{int(const.sum())} constant positions; expected layout has "
        f"{len(SYNC) + len(TRAILER) + len(HEADER)} (s = sync/trailer, h = header)",
    ]
    if len(frames) > 1:
        chance = FRAME_SYMBOLS / 16 ** (len(frames) - 1)
        lines.append(f"random frames would share about {chance:.2f} positions by chance")
    lines.append("headers:")
    width = max(map(len, names))
    for name, frame in zip(names, aligned, strict=True):
        header = " ".join(format_symbols(symbol[None]) for symbol in frame[HEADER])
        lines.append(f"  {name:{width}}  {header}")
    if len(frames) > 1:
        lines.append(
            f"pairwise differing positions (random frames ~{FRAME_SYMBOLS * 15 / 16:.0f}):"
        )
        for (i, a), (j, b) in itertools.combinations(enumerate(aligned), 2):
            lines.append(f"  {names[i]} vs {names[j]}: {symbol_distance(a, b)}")
    return lines


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: per-position analysis of ``.sym`` files."""
    parser = argparse.ArgumentParser(
        description="Find constant positions, headers and distances across FlexNet frames."
    )
    parser.add_argument("files", nargs="+", type=Path, help=".sym files, ideally one meter")
    args = parser.parse_args(argv)
    frames = [read_symbol_file(p) for p in args.files]
    print("\n".join(report(frames, [p.stem for p in args.files])))


if __name__ == "__main__":
    main()
