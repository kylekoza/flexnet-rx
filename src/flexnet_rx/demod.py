"""Demodulate one FlexNet Normal-mode burst into 16-QAM symbols.

FlexNet's Normal uplink mode, which Sensus's 2009 modulation specification calls
"7-FSK, a 16 QAM signal", is frequency modulation whose modulating waveform is a ~2011 Hz
subcarrier carrying one 16-QAM symbol per subcarrier cycle. The steps are:

1. Mix the burst to 0 Hz and decimate to about 240 kHz.
2. Low-pass to the channel, then FM-discriminate and low-pass again.
3. Fit the subcarrier frequency on the bare-tone preamble (the first ~12 ms).
4. Mix the discriminator output down by that frequency, rotating the preamble onto +I.
5. Integrate and dump over one subcarrier period per symbol, sweeping the start phase.
6. At each phase fit a scale and rotation to a 16-QAM grid; keep the tightest fit.
7. Slice I and Q to the levels -3, -1, +1, +3.

The fit score is the RMS distance to the nearest grid point in units of half the grid
spacing: a clean real frame scores about 0.13-0.15, a wrong timing phase about 0.44 and
random points about 0.29.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from flexnet_rx.radio import CENTER_HZ, PREAMBLE_MS, RATE, freq_from_name, load_cf32
from flexnet_rx.symbols import (
    LEVELS,
    ComplexArray,
    FloatArray,
    IntArray,
    IQArray,
    SymbolArray,
    format_symbols,
)

_LEVEL_EDGES = np.array([-2.0, 0.0, 2.0])


@dataclass(frozen=True)
class DemodConfig:
    """Demodulator settings; the defaults suit Normal mode from an RTL-SDR at 2.4 Msps.

    Attributes:
        target_rate: Sample rate to decimate to before discriminating.
        channel_bw_hz: Cut-off of the low-pass before the discriminator.
        post_bw_hz: Cut-off of the low-pass after it.
        preamble_ms: Start and end of the window used to fit the subcarrier.
        data_start_ms: Where symbol integration starts, after the preamble.
        subcarrier_range_hz: Lowest and highest subcarrier frequency searched.
        subcarrier_step_hz: Step of that search.
        phase_steps: Symbol-timing phases tried across one subcarrier period.
        rotation_steps: Grid rotations tried across 90 degrees.
        rotation_start_deg: Start of the 90-degree rotation window. 16-QAM looks the same
            every 90 degrees, so the window decides the labelling; a frame whose rotation
            falls near an edge could be labelled differently from its neighbours. Real
            frames sit near 64 degrees, so the default window of 18.75-108.75 degrees puts
            them in the middle, on the same 3.75-degree lattice as a window from 0.
    """

    target_rate: float = 240e3
    channel_bw_hz: float = 40e3
    post_bw_hz: float = 14e3
    preamble_ms: tuple[float, float] = (0.85, 11.0)
    data_start_ms: float = PREAMBLE_MS
    subcarrier_range_hz: tuple[float, float] = (2000.0, 2022.0)
    subcarrier_step_hz: float = 0.02
    phase_steps: int = 32
    rotation_steps: int = 24
    rotation_start_deg: float = 18.75


@dataclass(frozen=True)
class Demodulated:
    """The result of demodulating one burst.

    Attributes:
        symbols: Hard decisions as an ``(n, 2)`` array of (I, Q) level indexes.
        points: The normalised constellation points the decisions were made from.
        subcarrier_hz: Fitted subcarrier frequency, which is also the symbol rate.
        score: RMS distance to the grid in half-spacings; lower is cleaner.
        phase: Chosen symbol-timing phase, as a fraction of one period.
        rotation_deg: Grid rotation removed after the preamble alignment.
    """

    symbols: SymbolArray
    points: ComplexArray
    subcarrier_hz: float
    score: float
    phase: float
    rotation_deg: float

    @property
    def text(self) -> str:
        """The symbols as one line of digit pairs."""
        return format_symbols(self.symbols)


def discriminate(
    iq: IQArray, offset_hz: float, rate: float, cfg: DemodConfig
) -> tuple[FloatArray, float]:
    """FM-discriminate the burst and trim it to where the signal is present.

    Args:
        iq: Complex samples of one burst with some padding either side.
        offset_hz: Burst frequency minus the tuned frequency.
        rate: Sample rate of ``iq``.
        cfg: Demodulator settings.

    Returns:
        The instantaneous frequency in Hz, trimmed to the burst, and its sample rate.

    Raises:
        ValueError: If the burst is too short to hold a preamble and data.
    """
    if len(iq) < rate * 1e-3 * (cfg.data_start_ms + 5):
        raise ValueError("burst too short for a preamble and data")
    factor = max(1, round(rate / cfg.target_rate))
    afs = rate / factor
    n = np.arange(len(iq))
    baseband = iq * np.exp(-2j * np.pi * offset_hz * n / rate)
    baseband = signal.resample_poly(baseband, 1, factor)
    lowpass = signal.butter(5, cfg.channel_bw_hz, fs=afs, output="sos")
    baseband = signal.sosfiltfilt(lowpass, baseband)
    freq = np.diff(np.unwrap(np.angle(baseband))) * afs / (2 * np.pi)
    post = signal.butter(4, cfg.post_bw_hz, fs=afs, output="sos")
    freq = signal.sosfiltfilt(post, freq)
    amplitude = np.abs(baseband)
    present = np.flatnonzero(amplitude > 0.3 * np.percentile(amplitude, 99))
    if present.size == 0:
        raise ValueError("no signal in the burst")
    edge = round(125e-6 * afs)
    start, end = present[0] + edge, present[-1] - edge
    if end - start < (cfg.data_start_ms + 5) * 1e-3 * afs:
        raise ValueError("burst too short for a preamble and data")
    return np.asarray(freq[start:end], dtype=np.float64), afs


def fit_subcarrier(freq: FloatArray, afs: float, cfg: DemodConfig) -> tuple[float, ComplexArray]:
    """Find the subcarrier on the preamble and mix the discriminator output down by it.

    Args:
        freq: Discriminator output from :func:`discriminate`.
        afs: Its sample rate.
        cfg: Demodulator settings.

    Returns:
        The subcarrier frequency and the complex baseband around it, rotated so the
        preamble lies on the positive real axis.
    """
    lo, hi = (int(ms * 1e-3 * afs) for ms in cfg.preamble_ms)
    freq = freq - freq[lo:hi].mean()
    n = np.arange(lo, hi)
    candidates = np.arange(*cfg.subcarrier_range_hz, cfg.subcarrier_step_hz)
    energy = np.abs(np.exp(-2j * np.pi * np.outer(candidates, n) / afs) @ freq[lo:hi])
    fc = float(candidates[int(np.argmax(energy))])
    z = freq * np.exp(-2j * np.pi * fc * np.arange(len(freq)) / afs)
    z = z * np.exp(-1j * np.angle(z[lo:hi].mean()))
    return fc, z


def integrate_and_dump(
    z: ComplexArray, fc: float, afs: float, phase: float, start_ms: float
) -> ComplexArray:
    """Average ``z`` over consecutive subcarrier periods.

    Args:
        z: Complex baseband around the subcarrier.
        fc: Subcarrier frequency; one symbol lasts ``1 / fc``.
        afs: Sample rate of ``z``.
        phase: Offset of the first symbol boundary, as a fraction of a period.
        start_ms: Where the first symbol may start.

    Returns:
        One complex value per whole symbol period.
    """
    period = afs / fc
    first = start_ms * 1e-3 * afs + phase * period
    count = int((len(z) - first) / period)
    edges = first + np.arange(count + 1) * period
    lo = np.ceil(edges[:-1]).astype(int)
    hi = np.floor(edges[1:]).astype(int)
    csum = np.concatenate([[0], np.cumsum(z)])
    return np.asarray((csum[hi] - csum[lo]) / (hi - lo), dtype=np.complex128)


def fit_grid(
    points: ComplexArray, rotation_steps: int, rotation_start_deg: float = 0.0
) -> tuple[float, float, float]:
    """Fit a rotation and scale that best place ``points`` on the 16-QAM grid.

    The scale comes from the mean power (10 for a unit-spaced grid), the rotation from a
    search over 90 degrees.

    Args:
        points: Integrated symbol values.
        rotation_steps: Rotations tried across 90 degrees.
        rotation_start_deg: Start of the 90-degree window searched.

    Returns:
        ``(score, rotation_rad, gain)``: RMS distance to the nearest grid point in
        half-spacings, and the rotation and gain that achieve it.
    """
    gain = float(np.sqrt(np.mean(np.abs(points) ** 2) / 10))
    if gain == 0:
        return float("inf"), 0.0, 1.0
    best_score, best_rotation = float("inf"), 0.0
    start = np.radians(rotation_start_deg)
    for rotation in start + np.linspace(0, np.pi / 2, rotation_steps, endpoint=False):
        q = points * np.exp(-1j * rotation) / gain
        nearest = LEVELS[_level_index(q.real)] + 1j * LEVELS[_level_index(q.imag)]
        score = float(np.sqrt(np.mean(np.abs(q - nearest) ** 2)) / 2)
        if score < best_score:
            best_score, best_rotation = score, float(rotation)
    return best_score, best_rotation, gain


def _level_index(values: FloatArray) -> IntArray:
    """Return the index 0-3 of the nearest of -3, -1, +1, +3; ties go to the lower level."""
    return np.asarray(np.searchsorted(_LEVEL_EDGES, values, side="left"), dtype=np.int64)


def demodulate(
    iq: IQArray,
    offset_hz: float,
    rate: float = RATE,
    cfg: DemodConfig | None = None,
) -> Demodulated:
    """Demodulate one Normal-mode burst.

    Args:
        iq: Complex samples of one burst with some padding either side.
        offset_hz: Burst frequency minus the tuned frequency.
        rate: Sample rate of ``iq``.
        cfg: Demodulator settings; the defaults suit Normal mode.

    Returns:
        The decisions and the fit that produced them.
    """
    cfg = cfg or DemodConfig()
    freq, afs = discriminate(iq, offset_hz, rate, cfg)
    fc, z = fit_subcarrier(freq, afs, cfg)
    trials = []
    for start in np.linspace(0, 1, cfg.phase_steps, endpoint=False):
        points = integrate_and_dump(z, fc, afs, float(start), cfg.data_start_ms)
        trials.append(
            (
                fit_grid(points[2:-1], cfg.rotation_steps, cfg.rotation_start_deg),
                float(start),
                points,
            )
        )
    (score, rotation, gain), phase, points = min(trials, key=lambda t: t[0][0])
    normalised = points * np.exp(-1j * rotation) / gain
    symbols = np.stack([_level_index(normalised.real), _level_index(normalised.imag)], axis=1)
    return Demodulated(
        symbols=symbols,
        points=normalised,
        subcarrier_hz=fc,
        score=score,
        phase=phase,
        rotation_deg=float(np.degrees(rotation)),
    )


def plot_constellation(result: Demodulated, title: str, path: Path) -> None:
    """Save a constellation plot; needs the optional ``plot`` extra (matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(result.points.real, result.points.imag, ".", ms=5)
    for level in LEVELS:
        ax.axhline(level, lw=0.3, color="tab:red")
        ax.axvline(level, lw=0.3, color="tab:red")
    ax.set_aspect("equal")
    ax.set_title(f"{title}\nscore {result.score:.3f}, {len(result.symbols)} symbols")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: demodulate ``.cf32`` bursts into ``.sym`` files."""
    parser = argparse.ArgumentParser(
        description="Demodulate FlexNet Normal-mode bursts into 16-QAM symbol files."
    )
    parser.add_argument("bursts", nargs="+", type=Path, help=".cf32 files from flexnet-capture")
    parser.add_argument("--center-hz", type=float, default=CENTER_HZ, help="tuned frequency")
    parser.add_argument("--rate", type=float, default=RATE, help="sample rate of the files")
    parser.add_argument(
        "--freq-hz", type=float, help="burst frequency, if it is not in the file name"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("syms"), help="where .sym go")
    parser.add_argument("--plot", action="store_true", help="also save constellation PNGs")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.bursts:
        freq = args.freq_hz or freq_from_name(path)
        if freq is None:
            parser.error(f"{path.name}: no frequency in the name; pass --freq-hz")
        try:
            result = demodulate(load_cf32(path), freq - args.center_hz, args.rate)
        except ValueError as exc:
            print(f"{path.name}: skipped ({exc})")
            continue
        out = args.out_dir / f"{path.stem}.sym"
        out.write_text(result.text + "\n")
        print(
            f"{path.name}: {len(result.symbols)} symbols, score {result.score:.3f}, "
            f"subcarrier {result.subcarrier_hz:.2f} Hz -> {out}"
        )
        if args.plot:
            plot_constellation(result, path.stem, out.with_suffix(".png"))


if __name__ == "__main__":
    main()
