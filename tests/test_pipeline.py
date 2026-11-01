"""End-to-end and unit tests on synthetic data; no real captures are needed or included."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from flexnet_rx import capture, demod, fec, frame, idsearch, radio, randomness, synth
from flexnet_rx.symbols import (
    bits_to_values,
    format_symbols,
    parse_symbols,
    symbol_values,
    values_to_symbols,
)


def _rotations(symbols: np.ndarray) -> list[np.ndarray]:
    """The four 90-degree rotations of a symbol array, as index pairs."""
    out = [symbols]
    for _ in range(3):
        i, q = out[-1][:, 0], out[-1][:, 1]
        out.append(np.stack([3 - q, i], axis=1))  # multiply by j on the grid
    return out


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def test_symbol_text_round_trip() -> None:
    text = "0332211203"
    assert format_symbols(parse_symbols(text)) == text
    with pytest.raises(ValueError):
        parse_symbols("0334")
    with pytest.raises(ValueError):
        parse_symbols("033")


def test_demodulates_synthetic_burst(rng: np.random.Generator) -> None:
    sent = synth.example_frame(rng)
    iq = synth.synth_burst(sent, rng, snr_db=25)
    result = demod.demodulate(iq, offset_hz=-438e3)
    assert abs(result.subcarrier_hz - radio.SUBCARRIER_HZ) < 0.5
    assert result.score < 0.15
    assert len(result.symbols) == len(sent)
    assert max((r == sent).all(axis=1).mean() for r in _rotations(result.symbols)) >= 0.99


def test_demod_rejects_a_click(rng: np.random.Generator) -> None:
    iq = (rng.normal(size=20000) + 1j * rng.normal(size=20000)).astype(np.complex64) * 0.01
    iq[10000:10500] += 1
    with pytest.raises(ValueError):
        demod.demodulate(iq, 0.0)


def test_freq_from_name() -> None:
    assert radio.freq_from_name("20261101T020153.420_901.0623MHz_63ms.cf32") == pytest.approx(
        901.0623e6
    )
    assert radio.freq_from_name("burst.cf32") is None


def test_capture_logs_and_saves_one_burst(tmp_path: Path, rng: np.random.Generator) -> None:
    rate = 240_000
    burst = synth.synth_burst(
        synth.example_frame(rng), rng, rate=rate, offset_hz=40e3, snr_db=30, pad_ms=0
    )
    noise_ms = 2500
    noise = (rng.normal(size=noise_ms * 240) + 1j * rng.normal(size=noise_ms * 240)) * 0.02
    sig = np.concatenate([noise, burst * 0.5 + noise[: len(burst)], noise[:24000]])
    u8 = np.clip(np.stack([sig.real, sig.imag], axis=1).ravel() * 100 + 127.5, 0, 255)
    stream = io.BytesIO(u8.astype(np.uint8).tobytes())
    cfg = capture.CaptureConfig(center_hz=901.5e6, rate=rate, out_dir=tmp_path)
    capture.run(cfg, stream, clock=lambda: 1_790_000_000.0)
    rows = (tmp_path / "bursts.csv").read_text().splitlines()
    assert rows[0] == capture.CSV_HEADER.strip()
    assert len(rows) == 2
    fields = dict(zip(rows[0].split(","), rows[1].split(","), strict=True))
    assert abs(int(fields["duration_ms"]) - len(burst) / rate * 1000) <= 3
    assert abs(float(fields["peak_offset_khz"]) - 40) < 6
    assert (tmp_path / fields["file"]).exists()


def test_burst_stamp_rounds_milliseconds() -> None:
    assert capture.burst_stamp(0.9996) == "19700101T000001.000"


def test_frame_layout_counts() -> None:
    assert len(frame.SYNC) + len(frame.HEADER) + len(frame.PAYLOAD) + len(frame.TRAILER) == 100
    short = synth.example_frame(np.random.default_rng(0))[1:]
    split = frame.split_frame(short)
    assert (split.sync[0] == frame.MISSING).all()
    assert (split.trailer == frame.TRAILER_SYMBOLS).all()


def test_constant_positions_find_the_header(rng: np.random.Generator) -> None:
    header = rng.integers(0, 4, size=(8, 2))
    frames = [synth.example_frame(rng, header) for _ in range(6)]
    const = frame.constant_positions([frame.align_to_trailer(f) for f in frames])
    assert set(np.flatnonzero(const)) == {*frame.SYNC, *frame.HEADER, *frame.TRAILER}


def test_randomness_passes_random_and_fails_structure(rng: np.random.Generator) -> None:
    random_payloads = [rng.integers(0, 16, size=88) for _ in range(14)]
    result = randomness.measure(random_payloads)
    assert result.chi_square < 40
    assert abs(result.bit_balance_z) < 4
    counter = [np.arange(k, k + 88) % 16 for k in range(14)]
    structured = randomness.measure(counter)
    assert structured.longest_shared_run > 20
    assert structured.max_abs_lag_corr > 0.5


def test_longest_common_run() -> None:
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([9, 3, 4, 5, 9, 1])
    assert randomness.longest_common_run(a, b) == 3


def test_viterbi_inverts_encode(rng: np.random.Generator) -> None:
    bits = rng.integers(0, 2, size=200)
    for g1, g2 in [(7, 5), (5, 7), (6, 7), (3, 5)]:
        coded = fec.encode(bits, g1, g2)
        coded[[10, 90]] ^= 1
        decoded, errors = fec.viterbi(coded, g1, g2)
        assert errors == 2
        assert (decoded == bits).all()


def test_gf2_search_finds_a_planted_code(rng: np.random.Generator) -> None:
    labelling = rng.permutation(16)
    order = np.arange(352).reshape(8, 44).T.ravel()
    frames = [
        symbol_values(synth.coded_frame(rng.integers(0, 2, 176), (6, 5), order, labelling))
        for _ in range(2)
    ]
    found = fec.search(frames, polys=[(6, 5), (7, 5), (3, 7)], skips=(0,))
    planted = next(c for c in found if c.interleaver == "b8x44" and c.polys == (6, 5))
    assert planted.dimension > 0
    noise = [rng.integers(0, 16, size=88) for _ in range(2)]
    noise_found = fec.search(noise, polys=[(6, 5)], skips=(0,))
    assert next(c for c in noise_found if c.interleaver == "b8x44").dimension == 0


def test_idsearch_finds_a_planted_bcd_number(rng: np.random.Generator) -> None:
    number = 12345678
    bits = np.array(idsearch.encodings(number)["bcd"])
    frame_bits = rng.integers(0, 2, size=400)
    frame_bits[120 : 120 + len(bits)] = bits
    whole = values_to_symbols(bits_to_values(frame_bits))
    whole[frame.TRAILER] = frame.TRAILER_SYMBOLS
    clipped = whole[1:]  # a weak meter's burst often loses its first symbol
    hits, expected = idsearch.search({"whole": whole, "clipped": clipped}, number)
    natural = {
        h.frame
        for h in hits
        if h.encoding == "bcd"
        and h.mapping.order == (0, 1, 2, 3)
        and not h.mapping.swap
        and not (h.mapping.negate_i or h.mapping.negate_q)
        and not (h.mapping.gray_i or h.mapping.gray_q)
        and h.offset == 120
    }
    assert natural == {"whole", "clipped"}
    assert expected < 1


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    synth.main([str(tmp_path / "bursts"), "--count", "3"])
    files = sorted((tmp_path / "bursts").glob("*.cf32"))
    demod.main([*map(str, files), "--out-dir", str(tmp_path / "syms")])
    syms = sorted((tmp_path / "syms").glob("*.sym"))
    assert len(syms) == 3
    frame.main(list(map(str, syms)))
    randomness.main([*map(str, syms), "--trials", "20"])
    out = capsys.readouterr().out
    assert "12 constant positions" in out
    assert "chi_square" in out


def _u8_stream(sig: np.ndarray) -> io.BytesIO:
    u8 = np.clip(np.stack([sig.real, sig.imag], axis=1).ravel() * 100 + 127.5, 0, 255)
    return io.BytesIO(u8.astype(np.uint8).tobytes())


def _noise(rng: np.random.Generator, ms: int, rate: int = 240_000) -> np.ndarray:
    n = ms * rate // 1000
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.02


def test_capture_survives_a_silent_first_block(tmp_path: Path, rng: np.random.Generator) -> None:
    burst = synth.synth_burst(synth.example_frame(rng), rng, rate=240_000, offset_hz=40e3, pad_ms=0)
    silence = np.full(240, -127.5 / 100 * (1 + 1j))  # 1 ms of u8 zeros
    sig = np.concatenate([silence, _noise(rng, 2500), burst * 0.5, _noise(rng, 100)])
    cfg = capture.CaptureConfig(center_hz=901.5e6, rate=240_000, out_dir=tmp_path)
    capture.run(cfg, _u8_stream(sig), clock=lambda: 0.0)
    assert len((tmp_path / "bursts.csv").read_text().splitlines()) == 2


def test_capture_saves_a_burst_cut_off_by_the_end(tmp_path: Path, rng: np.random.Generator) -> None:
    burst = synth.synth_burst(synth.example_frame(rng), rng, rate=240_000, offset_hz=40e3, pad_ms=0)
    sig = np.concatenate([_noise(rng, 2500), burst * 0.5])
    cfg = capture.CaptureConfig(center_hz=901.5e6, rate=240_000, out_dir=tmp_path)
    capture.run(cfg, _u8_stream(sig), clock=lambda: 0.0)
    assert len((tmp_path / "bursts.csv").read_text().splitlines()) == 2


def test_alignment_finds_the_trailer_before_a_stray_symbol(rng: np.random.Generator) -> None:
    whole = synth.example_frame(rng)
    stray = np.concatenate([whole, [[0, 0]]])
    assert (frame.align_to_trailer(stray) == whole).all()


def test_fec_ignores_unknowns_no_equation_reads(rng: np.random.Generator) -> None:
    frames = [rng.integers(0, 15, size=88) for _ in range(14)]
    frames[0][0] = 15  # a value seen only in a symbol the skip leaves out
    found = fec.search(frames, polys=[(7, 5)], skips=(1,))
    assert max(c.dimension for c in found) == 0
