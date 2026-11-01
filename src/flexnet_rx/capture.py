"""Record every radio burst from an ``rtl_sdr`` IQ stream.

Reads the unsigned 8-bit interleaved IQ that ``rtl_sdr`` writes to stdout, tracks a running
noise floor in 1 ms blocks, and treats anything a set margin above the floor as a burst.
Every burst gets one row in ``bursts.csv`` (time, duration, peak level, peak frequency) and,
until a disk budget is reached, its raw IQ with padding either side as a ``.cf32`` file
(complex64, little-endian, at the capture sample rate).

File names carry the burst's peak frequency, which ``flexnet-demod`` reads back::

    20261101T020153.420_901.0623MHz_63ms.cf32

Example:
    rtl_sdr -f 901500000 -s 2400000 - | flexnet-capture 901500000 2400000 bursts/

Burst times come from the sample count since the stream started, so they stay accurate
when the input is a file played faster than real time, and drift only if the dongle drops
samples.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np

from flexnet_rx.radio import PAD_MS, burst_name, save_cf32
from flexnet_rx.symbols import IQArray

CSV_HEADER = "utc,epoch,duration_ms,peak_db_over_floor,peak_offset_khz,peak_mhz,file\n"


@dataclass(frozen=True)
class CaptureConfig:
    """Settings for the burst recorder.

    Attributes:
        center_hz: Frequency the dongle is tuned to.
        rate: Sample rate in samples per second; must be a multiple of 1000.
        out_dir: Directory for ``bursts.csv`` and the ``.cf32`` files.
        threshold_db: Margin above the noise floor that starts a burst.
        pad_ms: Milliseconds of IQ kept before and after each burst; also how long the
            signal must stay quiet to end one.
        max_burst_ms: Longest burst saved; anything longer is saved truncated.
        min_burst_ms: Bursts shorter than this are ignored.
        max_saved_bytes: Stop saving IQ (but keep logging) past this many bytes.
        warmup_ms: Time to let the noise floor settle before detecting anything.
    """

    center_hz: float
    rate: int
    out_dir: Path
    threshold_db: float = 10.0
    pad_ms: int = PAD_MS
    max_burst_ms: int = 2000
    min_burst_ms: int = 2
    max_saved_bytes: int = 20 * 1024**3
    warmup_ms: int = 2000


def peak_offset_hz(x: IQArray, rate: float, dc_guard_hz: float = 5000.0) -> float:
    """Return the frequency offset of the strongest component in ``x``.

    Averages 4096-point Hann-windowed power spectra and ignores the dongle's DC spike.

    Args:
        x: Complex baseband samples.
        rate: Their sample rate.
        dc_guard_hz: Half-width of the band around 0 Hz to ignore.

    Returns:
        The offset from the tuned frequency, in Hz.
    """
    nfft = 4096
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    segments = x[: (len(x) // nfft) * nfft].reshape(-1, nfft)
    spectrum = np.fft.fftshift(np.fft.fft(segments * np.hanning(nfft), axis=1), axes=1)
    power = (np.abs(spectrum) ** 2).mean(axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / rate))
    power[np.abs(freqs) < dc_guard_hz] = 0
    return float(freqs[int(np.argmax(power))])


def iter_blocks(stream: BinaryIO, samples: int) -> Iterator[IQArray]:
    """Yield successive blocks of ``samples`` complex samples from rtl_sdr's u8 output.

    Args:
        stream: A binary stream of interleaved unsigned 8-bit I and Q.
        samples: Complex samples per block.

    Yields:
        Each block as complex64 with its mean removed. A short final block is dropped.
    """
    nbytes = samples * 2
    while len(raw := stream.read(nbytes)) == nbytes:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 127.5).view(np.complex64)
        yield x - x.mean()


def burst_stamp(t: float) -> str:
    """Format a POSIX time as ``YYYYmmddTHHMMSS.mmm`` in UTC."""
    ms = round(t * 1000)
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime(ms // 1000)) + f".{ms % 1000:03d}"


class BurstRecorder:
    """Detect bursts in 1 ms blocks, log each one, and save its IQ.

    Feed it blocks with :meth:`feed`; it writes to ``log`` and ``cfg.out_dir`` as each
    burst ends.
    """

    def __init__(self, cfg: CaptureConfig, log: TextIO, start_time: float) -> None:
        """Set up a recorder.

        Args:
            cfg: Capture settings.
            log: Open text file for CSV rows; the header is written if it is empty.
            start_time: POSIX time of the first sample.
        """
        if cfg.rate % 1000:
            raise ValueError("sample rate must be a multiple of 1000")
        self.cfg = cfg
        self.log = log
        self.start_time = start_time
        self.blocks_seen = 0
        self.saved_bytes = 0
        self.floor_db: float | None = None
        self.history: deque[IQArray] = deque(maxlen=cfg.pad_ms)
        self._burst: list[IQArray] | None = None
        self._pre = 0
        self._length = 0
        self._quiet = 0
        self._peak_db = 0.0
        self._t0 = 0.0
        if log.tell() == 0:
            log.write(CSV_HEADER)

    def feed(self, x: IQArray) -> None:
        """Process one 1 ms block of complex samples."""
        cfg = self.cfg
        level_db = 10 * math.log10(float(np.vdot(x, x).real) / len(x) + 1e-9)
        self.blocks_seen += 1
        if self.floor_db is None:
            self.floor_db = level_db
        if self._burst is None:
            if self.blocks_seen <= cfg.warmup_ms:
                # Settle on whatever the noise is, even from a wildly wrong first block.
                self.floor_db = 0.99 * self.floor_db + 0.01 * level_db
            elif level_db < self.floor_db + cfg.threshold_db:
                self.floor_db = 0.999 * self.floor_db + 0.001 * level_db
            if level_db > self.floor_db + cfg.threshold_db and self.blocks_seen > cfg.warmup_ms:
                self._burst = [*self.history, x]
                self._pre = len(self.history)
                self._length = 1
                self._quiet = 0
                self._peak_db = level_db
                self._t0 = self.start_time + (self.blocks_seen - 1) / 1000
            else:
                self.history.append(x)
            return
        self._length += 1
        if len(self._burst) < cfg.max_burst_ms + 2 * cfg.pad_ms:
            self._burst.append(x)
        self._peak_db = max(self._peak_db, level_db)
        quiet = level_db < self.floor_db + cfg.threshold_db / 2
        self._quiet = self._quiet + 1 if quiet else 0
        if self._quiet >= cfg.pad_ms:
            self._finish()
        elif self._length > 2 * cfg.max_burst_ms:
            # Nothing transmits this long: the floor itself has risen (gain drift, a new
            # interferer). Save what there is and adopt the new level as the floor.
            self._finish()
            self.floor_db = level_db

    def close(self) -> None:
        """Finish a burst still open when the stream ends."""
        if self._burst is not None:
            self._finish()

    def _finish(self) -> None:
        """Log and save the burst that just ended, then reset for the next one."""
        cfg = self.cfg
        assert self._burst is not None and self.floor_db is not None
        duration = self._length - self._quiet
        burst, self._burst = self._burst, None
        self.history.clear()
        if duration < cfg.min_burst_ms:
            return
        iq = np.concatenate(burst)
        block = cfg.rate // 1000
        body = iq[self._pre * block : (self._pre + min(duration, cfg.max_burst_ms)) * block]
        offset = peak_offset_hz(body, cfg.rate)
        stamp = burst_stamp(self._t0)
        name = burst_name(stamp, cfg.center_hz + offset, duration)
        if self.saved_bytes < cfg.max_saved_bytes:
            self.saved_bytes += save_cf32(cfg.out_dir / name, iq)
        else:
            name = ""
        self.log.write(
            f"{stamp},{self._t0:.3f},{duration},{self._peak_db - self.floor_db:.1f},"
            f"{offset / 1e3:.1f},{(cfg.center_hz + offset) / 1e6:.4f},{name}\n"
        )
        self.log.flush()


def run(cfg: CaptureConfig, stream: BinaryIO, clock: Callable[[], float] = time.time) -> int:
    """Record bursts from ``stream`` until it ends.

    Args:
        cfg: Capture settings.
        stream: rtl_sdr's u8 IQ output.
        clock: Source of the start time, for tests.

    Returns:
        The number of 1 ms blocks processed.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.out_dir / "bursts.csv").open("a", encoding="utf-8") as log:
        recorder = BurstRecorder(cfg, log, clock())
        for block in iter_blocks(stream, cfg.rate // 1000):
            recorder.feed(block)
        recorder.close()
        return recorder.blocks_seen


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: ``flexnet-capture CENTER_HZ RATE OUT_DIR``."""
    parser = argparse.ArgumentParser(
        description="Log and save every burst in an rtl_sdr u8 IQ stream read from stdin.",
        epilog="Example: rtl_sdr -f 901500000 -s 2400000 - | flexnet-capture 901500000 "
        "2400000 bursts/",
    )
    parser.add_argument("center_hz", type=float, help="frequency rtl_sdr is tuned to")
    parser.add_argument("rate", type=int, help="rtl_sdr sample rate (multiple of 1000)")
    parser.add_argument("out_dir", type=Path, help="directory for bursts.csv and .cf32 files")
    parser.add_argument("--threshold-db", type=float, default=CaptureConfig.threshold_db)
    parser.add_argument("--pad-ms", type=int, default=CaptureConfig.pad_ms)
    parser.add_argument("--max-burst-ms", type=int, default=CaptureConfig.max_burst_ms)
    parser.add_argument(
        "--max-saved-gb",
        type=float,
        default=CaptureConfig.max_saved_bytes / 1024**3,
        help="stop saving IQ past this (GiB)",
    )
    args = parser.parse_args(argv)
    cfg = CaptureConfig(
        center_hz=args.center_hz,
        rate=args.rate,
        out_dir=args.out_dir,
        threshold_db=args.threshold_db,
        pad_ms=args.pad_ms,
        max_burst_ms=args.max_burst_ms,
        max_saved_bytes=int(args.max_saved_gb * 1024**3),
    )
    run(cfg, sys.stdin.buffer)


if __name__ == "__main__":
    main()
