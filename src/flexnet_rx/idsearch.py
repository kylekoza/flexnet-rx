"""Look for a known number, such as a printed meter ID, in demodulated frames.

If a meter sent its ID in the clear, it would appear as some bit pattern under some
mapping of 16-QAM symbols to bits. This tool tries every mapping in a broad family and
every common encoding of the number, at every bit offset:

* Mappings: I and Q in either order, either axis negated, Gray or natural coding per axis,
  and all 24 orders of the 4 bits within a symbol (768 mappings).
* Encodings: BCD of the decimal digits; plain binary at its minimum width and at 24 and 32
  bits; each also byte-swapped, least-significant-bit first within bytes, and reversed.

A hit is printed with the mapping that produced it. Short patterns can match by chance, so
the tool also prints how many chance hits to expect.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from flexnet_rx.frame import MISSING, align_to_trailer
from flexnet_rx.symbols import IntArray, SymbolArray, read_symbol_file

GRAY = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
NATURAL = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


def _bits(value: int, width: int) -> list[int]:
    """Return ``value`` as ``width`` bits, most significant first."""
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def encodings(number: int) -> dict[str, list[int]]:
    """Return the bit patterns to look for, keyed by a description.

    Args:
        number: A non-negative integer, e.g. a printed serial number.

    Returns:
        Bit patterns for BCD and binary encodings and their byte-order variants.

    Raises:
        ValueError: If ``number`` is negative.
    """
    if number < 0:
        raise ValueError("number must be non-negative")
    digits = str(number)
    if len(digits) % 2:
        digits = "0" + digits
    base = {"bcd": [b for d in digits for b in _bits(int(d), 4)]}
    for width in sorted({max(1, number.bit_length()), 24, 32}):
        if number.bit_length() <= width:
            base[f"bin{width}"] = _bits(number, width)
    patterns = {}
    for name, bits in base.items():
        patterns[name] = bits
        patterns[f"{name}-reversed"] = bits[::-1]
        if len(bits) % 8 == 0:
            octets = [bits[i : i + 8] for i in range(0, len(bits), 8)]
            patterns[f"{name}-byteswap"] = [b for o in octets[::-1] for b in o]
            patterns[f"{name}-lsbfirst"] = [b for o in octets for b in o[::-1]]
    return patterns


@dataclass(frozen=True)
class Mapping:
    """One way of turning 16-QAM symbols into 4 bits.

    Attributes:
        swap: Read Q before I.
        negate_i: Reverse the I level order.
        negate_q: Reverse the Q level order.
        gray_i: Gray-code I rather than natural binary.
        gray_q: Gray-code Q rather than natural binary.
        order: The order the four bits are emitted in.
    """

    swap: bool
    negate_i: bool
    negate_q: bool
    gray_i: bool
    gray_q: bool
    order: tuple[int, ...]

    def apply(self, symbols: SymbolArray) -> IntArray:
        """Return the bit stream for ``symbols`` under this mapping."""
        i, q = (symbols[:, 1], symbols[:, 0]) if self.swap else (symbols[:, 0], symbols[:, 1])
        if self.negate_i:
            i = 3 - i
        if self.negate_q:
            q = 3 - q
        bits = np.concatenate(
            [(GRAY if self.gray_i else NATURAL)[i], (GRAY if self.gray_q else NATURAL)[q]], axis=1
        )
        return np.asarray(bits[:, list(self.order)].ravel())


def mappings() -> Iterator[Mapping]:
    """Yield every mapping in the family the module docstring describes."""
    for flags in itertools.product((False, True), repeat=5):
        for order in itertools.permutations(range(4)):
            swap, negate_i, negate_q, gray_i, gray_q = flags
            yield Mapping(swap, negate_i, negate_q, gray_i, gray_q, order)


def _as_text(bits: IntArray | list[int]) -> str:
    """Render bits as a string of 0s and 1s, for fast substring search."""
    return "".join(map(str, bits))


class Hit(NamedTuple):
    """One place a pattern was found.

    Attributes:
        frame: Label of the frame.
        encoding: Which encoding of the number matched.
        offset: Bit offset of the match.
        mapping: The symbol-to-bit mapping that produced it.
    """

    frame: str
    encoding: str
    offset: int
    mapping: Mapping


def search(frames: dict[str, SymbolArray], number: int) -> tuple[list[Hit], float]:
    """Search frames for every encoding of ``number`` under every mapping.

    Frames are aligned on their trailer first, so a bit offset means the same place in
    every frame even when a burst lost leading symbols; missing symbols never match.

    Args:
        frames: Symbol arrays keyed by a label.
        number: The number to look for.

    Returns:
        Every occurrence, and the number of mapping-hits expected by chance.
    """
    patterns = {name: _as_text(bits) for name, bits in encodings(number).items()}
    all_mappings = list(mappings())
    hits = []
    expected = 0.0
    for label, symbols in frames.items():
        aligned = align_to_trailer(symbols)
        missing = np.repeat((aligned == MISSING).any(axis=1), 4)
        safe = np.where(aligned == MISSING, 0, aligned)
        nbits = int((~missing).sum())
        per_mapping = sum(max(0, nbits - len(p) + 1) / 2 ** len(p) for p in patterns.values())
        expected += per_mapping * len(all_mappings)
        for mapping in all_mappings:
            chars = mapping.apply(safe).astype(str)
            chars[missing] = "x"
            text = "".join(chars)
            for name, pattern in patterns.items():
                offset = text.find(pattern)
                while offset >= 0:
                    hits.append(Hit(label, name, offset, mapping))
                    offset = text.find(pattern, offset + 1)
    return hits, expected


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: ``flexnet-idsearch NUMBER FILE...``."""
    parser = argparse.ArgumentParser(
        description="Look for a known number (e.g. a printed meter ID) in FlexNet frames."
    )
    parser.add_argument("number", type=int, help="the number to look for, in decimal")
    parser.add_argument("files", nargs="+", type=Path, help=".sym files")
    args = parser.parse_args(argv)
    frames = {p.stem: read_symbol_file(p) for p in args.files}
    hits, expected = search(frames, args.number)
    by_place: dict[tuple[str, str, int], list[Mapping]] = {}
    recurring: dict[tuple[str, int, Mapping], set[str]] = {}
    for hit in hits:
        by_place.setdefault((hit.frame, hit.encoding, hit.offset), []).append(hit.mapping)
        recurring.setdefault((hit.encoding, hit.offset, hit.mapping), set()).add(hit.frame)
    for (label, name, offset), found in by_place.items():
        print(f"HIT {label}: {name} at bit {offset} under {len(found)} mapping(s), e.g. {found[0]}")
    best = max((len(v) for v in recurring.values()), default=0)
    print(
        f"{len(by_place)} distinct hits across {len(frames)} frames "
        f"(about {expected:.3f} mapping-hits expected by chance)."
    )
    print(
        f"Most frames sharing one hit (same encoding, offset and mapping): {best}. "
        "An ID sent in the clear would recur in every frame from that meter."
    )


if __name__ == "__main__":
    main()
