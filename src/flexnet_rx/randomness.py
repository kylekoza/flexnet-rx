"""Tests of whether frame payloads are distinguishable from random data.

An unencrypted meter report would show fixed fields, slowly changing counters and values
repeated between frames an hour apart. Ciphertext, or a strong per-frame whitener, shows
none of that. These tests measure the payloads of several frames from one meter and print
each result beside what random data of the same size gives, from a Monte Carlo baseline.

Passing every test does not identify a cipher; it only says nothing structured is visible.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from flexnet_rx.frame import load_values
from flexnet_rx.symbols import IntArray, values_to_bits


@dataclass(frozen=True)
class Results:
    """Randomness statistics for a set of payloads.

    Attributes:
        symbols: Total payload symbols measured.
        chi_square: Chi-square of the 16 symbol-value counts (15 degrees of freedom).
        ones_fraction: Share of 1 bits under the natural ``4*I + Q`` labelling.
        bit_balance_z: z-score of that share against one half.
        runs_z: z-score of the number of runs of equal bits (Wald-Wolfowitz).
        max_abs_lag_corr: Largest absolute correlation of symbol values at lags 1-12.
        longest_shared_run: Longest run of symbols that appears in two different frames.
        mean_entropy: Mean per-frame entropy of symbol values, bits per symbol.
    """

    symbols: int
    chi_square: float
    ones_fraction: float
    bit_balance_z: float
    runs_z: float
    max_abs_lag_corr: float
    longest_shared_run: int
    mean_entropy: float


def chi_square(values: IntArray, categories: int = 16) -> float:
    """Return the chi-square statistic of ``values`` against a uniform distribution."""
    return float(stats.chisquare(np.bincount(values, minlength=categories)).statistic)


def runs_z(bits: IntArray) -> float:
    """Return the Wald-Wolfowitz runs-test z-score of a bit sequence."""
    n = len(bits)
    ones = int(bits.sum())
    zeros = n - ones
    if ones == 0 or zeros == 0:
        return float("inf")
    runs = 1 + int((bits[1:] != bits[:-1]).sum())
    mean = 1 + 2 * ones * zeros / n
    var = (mean - 1) * (mean - 2) / (n - 1)
    return (runs - mean) / math.sqrt(var)


def lag_correlation(values: IntArray, lag: int) -> float:
    """Return the Pearson correlation of ``values`` with itself shifted by ``lag``."""
    a, b = values[:-lag].astype(float), values[lag:].astype(float)
    if a.std() == 0 or b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def longest_common_run(a: IntArray, b: IntArray) -> int:
    """Return the length of the longest run of symbols common to ``a`` and ``b``."""
    matcher = difflib.SequenceMatcher(None, a.tolist(), b.tolist(), autojunk=False)
    return int(matcher.find_longest_match().size)


def entropy(values: IntArray) -> float:
    """Return the empirical entropy of ``values`` in bits per symbol."""
    return float(stats.entropy(np.bincount(values), base=2))


def measure(payloads: Sequence[IntArray]) -> Results:
    """Run every test on a set of payloads given as symbol values 0-15.

    Args:
        payloads: One array of symbol values per frame.

    Returns:
        The statistics.
    """
    values = np.concatenate(list(payloads))
    bits = values_to_bits(values)
    ones = int(bits.sum())
    pairs = itertools.combinations(payloads, 2)
    return Results(
        symbols=len(values),
        chi_square=chi_square(values),
        ones_fraction=ones / len(bits),
        bit_balance_z=(ones - len(bits) / 2) / math.sqrt(len(bits) / 4),
        runs_z=runs_z(bits),
        max_abs_lag_corr=max(abs(lag_correlation(values, k)) for k in range(1, 13)),
        longest_shared_run=max((longest_common_run(a, b) for a, b in pairs), default=0),
        mean_entropy=float(np.mean([entropy(p) for p in payloads])),
    )


def baseline(
    shape: Sequence[int], trials: int, rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
    """Measure uniformly random payloads of the same shape many times.

    Args:
        shape: Length of each payload.
        trials: Monte Carlo trials.
        rng: Random source.

    Returns:
        For each statistic, its 2.5th and 97.5th percentiles over the trials.
    """
    runs = [measure([rng.integers(0, 16, size=n) for n in shape]) for _ in range(trials)]
    names = [f.name for f in dataclasses.fields(Results) if f.name != "symbols"]
    ranges = {}
    for name in names:
        lo, hi = np.percentile([getattr(r, name) for r in runs], [2.5, 97.5])
        ranges[name] = (float(lo), float(hi))
    return ranges


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: randomness tests on the payloads of ``.sym`` files."""
    parser = argparse.ArgumentParser(
        description="Test whether FlexNet payloads are distinguishable from random data."
    )
    parser.add_argument("files", nargs="+", type=Path, help=".sym files from one meter")
    parser.add_argument("--whole-frame", action="store_true", help="test all symbols, not payload")
    parser.add_argument("--trials", type=int, default=300, help="Monte Carlo baseline trials")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    payloads = load_values(args.files, args.whole_frame)
    results = measure(payloads)
    ranges = baseline([len(p) for p in payloads], args.trials, np.random.default_rng(args.seed))
    print(f"{len(payloads)} frames, {results.symbols} symbols")
    print(f"{'statistic':22} {'measured':>10}   random data, 95% range")
    for name, (lo, hi) in ranges.items():
        value = getattr(results, name)
        flag = "" if lo <= value <= hi else "   <-- outside"
        print(f"{name:22} {value:10.3f}   {lo:.3f} to {hi:.3f}{flag}")


if __name__ == "__main__":
    main()
