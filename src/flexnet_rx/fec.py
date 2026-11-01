"""Search frames for the rate-1/2, constraint-length-3 convolutional code.

The FlexNet modulation specification names an interleaved rate-1/2 K=3 convolutional code.
Finding it from symbols alone means guessing how each 16-QAM symbol maps to 4 bits, how
the bits are interleaved, and which generator polynomials are used. This module removes the
first guess entirely with linear algebra over GF(2):

* Each transmitted bit is ``f_j(s)``: bit ``j`` (0-3) of the unknown labelling of symbol
  value ``s`` (0-15). That is 64 unknown bits.
* After de-interleaving, the code's parity check ``g2*c1 + g1*c2 = 0`` holds at every
  position. Each check is a linear equation in the 64 unknowns.
* For the right interleaver and polynomials, the true labelling is a non-trivial solution.
  For a wrong one, hundreds of equations in 64 unknowns leave only trivial solutions.

``nullspace dimension > 0`` therefore flags a candidate. Labellings that ignore the symbol
(``f_j(s)`` constant) always satisfy linear checks and are quotiented out. Some
configurations show structure on any input; :func:`search` can run the same search on
random symbols so those artefacts can be discounted.

Polynomials are written with bit ``k`` as the coefficient of ``D**k``: bit 0 is the current
input bit.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flexnet_rx.frame import load_values
from flexnet_rx.symbols import IntArray

DEFAULT_POLYS = tuple(itertools.permutations((3, 5, 6, 7), 2))
"""Every ordered pair of distinct K=3 generator polynomials with a D**0 or D**2 tap."""


def _taps(g: int) -> list[int]:
    """Return the delays ``k`` whose coefficient of ``D**k`` is set in polynomial ``g``."""
    return [k for k in range(3) if (g >> k) & 1]


def encode(bits: IntArray, g1: int, g2: int) -> IntArray:
    """Encode bits with a rate-1/2, K=3 convolutional code.

    Args:
        bits: Information bits.
        g1: First generator polynomial, bit ``k`` = coefficient of ``D**k``.
        g2: Second generator polynomial.

    Returns:
        Coded bits, ``c1[0], c2[0], c1[1], c2[1], ...``.
    """
    c1, c2 = (
        np.convolve(bits, [(g >> k) & 1 for k in range(3)])[: len(bits)] % 2 for g in (g1, g2)
    )
    return np.stack([c1, c2], axis=1).ravel()


def viterbi(coded: IntArray, g1: int, g2: int) -> tuple[IntArray, int]:
    """Decode a rate-1/2, K=3 code by hard-decision Viterbi.

    The state holds the previous two input bits, most recent in bit 0.

    Args:
        coded: Received coded bits, pairs ``c1, c2``.
        g1: First generator polynomial.
        g2: Second generator polynomial.

    Returns:
        The decoded information bits and the number of coded-bit errors on the best path.
    """
    # outputs[reg] for the 3-bit register (input in bit 0, then the state)
    outputs = [((reg & g1).bit_count() & 1, (reg & g2).bit_count() & 1) for reg in range(8)]
    pairs = coded[: len(coded) // 2 * 2].reshape(-1, 2).tolist()
    inf = 10**9
    cost = [0, inf, inf, inf]
    back: list[list[tuple[int, int]]] = []
    for r1, r2 in pairs:
        new = [inf] * 4
        arg = [(0, 0)] * 4
        for state in range(4):
            if cost[state] >= inf:
                continue
            for u in (0, 1):
                reg = u | (state << 1)
                o1, o2 = outputs[reg]
                c = cost[state] + (o1 != r1) + (o2 != r2)
                if c < new[reg & 3]:
                    new[reg & 3] = c
                    arg[reg & 3] = (state, u)
        back.append(arg)
        cost = new
    state = int(np.argmin(cost))
    errors = cost[state]
    bits = []
    for arg in reversed(back):
        state, u = arg[state]
        bits.append(u)
    return np.array(bits[::-1], dtype=np.int64), errors


def _row_column(n: int) -> Iterator[tuple[int, IntArray]]:
    """Yield ``(rows, order)`` for every row/column interleaver of ``n`` items.

    Items are written in rows of ``n // rows`` and read out by column; both dimensions are
    at least 2.
    """
    for rows in range(2, n // 2 + 1):
        if n % rows == 0:
            yield rows, np.arange(n).reshape(rows, -1).T.ravel()


def interleavers(nsym: int, block_bits: int | None = None) -> Iterator[tuple[str, IntArray]]:
    """Yield candidate interleavers as (name, transmit order) pairs.

    Transmit order ``idx`` means transmitted bit ``k`` carries coded bit ``idx[k]``.

    Args:
        nsym: Symbols per frame (4 bits each).
        block_bits: Also try row/column interleavers within independent blocks of this
            many bits, e.g. 44 for 11-symbol blocks.

    Yields:
        ``none``; ``bRxC`` row/column bit interleavers over the frame or a span up to 16
        bits shorter; ``sRxC`` whole-symbol interleavers; ``blkRxC`` per-block ones.
    """
    n = nsym * 4
    yield "none", np.arange(n)
    for span in range(n, n - 17, -4):
        for rows, order in _row_column(span):
            yield f"b{rows}x{span // rows}", order
    for rows, order in _row_column(nsym):
        yield f"s{rows}x{nsym // rows}", (order[:, None] * 4 + np.arange(4)).ravel()
    if block_bits and n % block_bits == 0:
        for rows, block in _row_column(block_bits):
            full = np.concatenate([block + block_bits * b for b in range(n // block_bits)])
            yield f"blk{rows}x{block_bits // rows}", full


def gf2_rank(rows: Sequence[int]) -> int:
    """Return the rank over GF(2) of equations given as integer bit masks.

    Stops early once the rank equals the number of distinct unknowns the rows mention,
    which it cannot exceed.
    """
    mentioned = 0
    for row in rows:
        mentioned |= row
    limit = mentioned.bit_count()
    pivots: dict[int, int] = {}
    for row in rows:
        if len(pivots) >= limit:
            break
        while row:
            top = row.bit_length() - 1
            if top not in pivots:
                pivots[top] = row
                break
            row ^= pivots[top]
    return len(pivots)


def _unknowns(values: IntArray) -> IntArray:
    """Return, per transmitted bit, the id ``16*j + s`` of the labelling unknown it carries."""
    return 16 * np.tile(np.arange(4), len(values)) + np.repeat(values, 4)


def parity_equations(
    unknowns: IntArray, inverse: IntArray, skip: int, g1: int, g2: int
) -> list[int]:
    """Build the parity-check equations of one frame for one configuration.

    Args:
        unknowns: The frame's :func:`_unknowns`.
        inverse: ``argsort`` of the interleaver's transmit order.
        skip: Symbols to skip before the interleaved span starts; the span must fit.
        g1: First generator polynomial.
        g2: Second generator polynomial.

    Returns:
        The equations as bit masks over the 64 unknowns.
    """
    coded = unknowns[skip * 4 : skip * 4 + len(inverse)][inverse].tolist()
    c1, c2 = coded[0::2], coded[1::2]
    taps1, taps2 = _taps(g1), _taps(g2)
    equations = []
    for m in range(2, len(c1)):
        mask = 0
        for k in taps2:
            mask ^= 1 << c1[m - k]
        for k in taps1:
            mask ^= 1 << c2[m - k]
        equations.append(mask)
    return equations


def solution_dimension(equations: Sequence[int]) -> int:
    """Count non-trivial labellings that satisfy every equation.

    Only unknowns some equation mentions count: a symbol value that never falls where a
    parity check reads it is unconstrained, not evidence of a code.

    Args:
        equations: Parity checks as bit masks over the 64 unknowns.

    Returns:
        The nullspace dimension after removing labellings that ignore the symbol.
    """
    mentioned = 0
    for mask in equations:
        mentioned |= mask
    groups = sum(1 for j in range(4) if (mentioned >> (16 * j)) & 0xFFFF)
    constant_rows = [
        sum(1 << j for j in range(4) if ((mask >> (16 * j)) & 0xFFFF).bit_count() & 1)
        for mask in equations
    ]
    nullity = mentioned.bit_count() - gf2_rank(equations)
    return nullity - (groups - gf2_rank(constant_rows))


@dataclass(frozen=True)
class Candidate:
    """One searched configuration.

    Attributes:
        dimension: Non-trivial solution dimension; above 0 is a candidate code.
        interleaver: Interleaver name.
        skip: Leading symbols skipped.
        polys: Generator polynomials (g1, g2).
        equations: Number of equations solved.
    """

    dimension: int
    interleaver: str
    skip: int
    polys: tuple[int, int]
    equations: int


def search(
    frames: Sequence[IntArray],
    polys: Sequence[tuple[int, int]] = DEFAULT_POLYS,
    skips: Sequence[int] = (0, 1, 2),
    block_bits: int | None = None,
    differential: bool = False,
) -> list[Candidate]:
    """Search every configuration and return them, most promising first.

    Args:
        frames: Symbol values 0-15 per frame, ideally several frames from one meter.
            Frames of different lengths are trimmed at the start to the shortest, since
            frames are aligned on their end.
        polys: Polynomial pairs to try.
        skips: Leading symbols to try skipping.
        block_bits: Passed to :func:`interleavers`.
        differential: XOR each frame's equations with the first frame's, which cancels a
            whitening sequence that is the same in every frame (needs two or more frames).

    Returns:
        Every configuration tried, highest dimension first.
    """
    nsym = min(len(f) for f in frames)
    trimmed = [np.asarray(f[len(f) - nsym :]) for f in frames]
    unknowns = [_unknowns(f) for f in trimmed]
    results = []
    for name, order in interleavers(nsym, block_bits):
        inverse = np.argsort(order)
        for skip, (g1, g2) in itertools.product(skips, polys):
            if skip * 4 + len(order) > nsym * 4:
                continue
            per_frame = [parity_equations(u, inverse, skip, g1, g2) for u in unknowns]
            if differential:
                base = per_frame[0]
                equations = [
                    a ^ b for other in per_frame[1:] for a, b in zip(other, base, strict=True)
                ]
            else:
                equations = [e for eqs in per_frame for e in eqs]
            if not equations:
                continue
            dim = solution_dimension(equations)
            results.append(Candidate(dim, name, skip, (g1, g2), len(equations)))
    results.sort(key=lambda c: -c.dimension)
    return results


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: search ``.sym`` payloads for the convolutional code."""
    parser = argparse.ArgumentParser(
        description="Search FlexNet frames for a rate-1/2 K=3 convolutional code under any "
        "symbol labelling, by linear algebra over GF(2)."
    )
    parser.add_argument("files", nargs="+", type=Path, help=".sym files, ideally one meter")
    parser.add_argument("--whole-frame", action="store_true", help="use all symbols, not payload")
    parser.add_argument("--differential", action="store_true", help="cancel a fixed whitener")
    parser.add_argument("--block-bits", type=int, help="also try per-block interleavers")
    parser.add_argument("--top", type=int, default=10, help="candidates to print")
    parser.add_argument("--seed", type=int, default=0, help="seed for the random baseline")
    args = parser.parse_args(argv)
    values = load_values(args.files, args.whole_frame)
    rng = np.random.default_rng(args.seed)
    random_frames = [rng.integers(0, 16, size=len(v)) for v in values]
    found, noise = (
        search(f, block_bits=args.block_bits, differential=args.differential)
        for f in (values, random_frames)
    )
    artefacts = {(c.interleaver, c.skip, c.polys) for c in noise if c.dimension > 0}
    real = [c for c in found if (c.interleaver, c.skip, c.polys) not in artefacts]
    hits = [c for c in real if c.dimension > 0]
    print(
        f"{len(found)} configurations; {len(artefacts)} also show structure on random "
        f"symbols and are ignored; {len(hits)} candidates remain"
    )
    for c in real[: args.top]:
        print(
            f"  dim {c.dimension:2d}  {c.interleaver:10s} skip {c.skip}  polys {c.polys}  "
            f"({c.equations} equations)"
        )


if __name__ == "__main__":
    main()
