# flexnet-rx

Receive-only tools for Sensus FlexNet electric meters with an RTL-SDR. They record a meter's
900 MHz bursts, demodulate them into 16-QAM symbols, map the frame layout, and test the
payload. As far as we can find, this is the first published demodulator for FlexNet.

**What works:** clean symbols from every Normal-mode frame we captured, and a frame layout of
sync, a fixed per-meter header, a payload, and a trailer.

**What doesn't:** reading your usage. The payload is statistically random, and Sensus
documents AES-256 encryption for FlexNet. These tools stop there by design. If you want your
own data, your utility's portal or a supported integration is the route.

## Scope

- **Receive only.** Nothing here transmits.
- **No decryption.** There is no key recovery or cryptanalysis here, and contributions in that
  direction won't be accepted.
- **Your own meter.** The air is shared, so captures include your neighbours' meters. Use
  their frames only to tell protocol fields from per-meter ones, and don't publish their
  contents. The `.gitignore` keeps captures and symbol files out of version control.
- **Check your local rules** on receiving encrypted transmissions before you start.

## Install

Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
git clone <this repository> && cd flexnet-rx
uv sync --extra plot
```

Or with pip: `pip install -e '.[plot]'`. The `plot` extra adds matplotlib for constellation
plots and is optional.

## Try it without a meter

`flexnet-synth` writes synthetic bursts named like real captures, so you can run the whole
pipeline first:

```bash
uv run flexnet-synth demo/bursts --count 3
uv run flexnet-demod demo/bursts/*.cf32 --out-dir demo/syms --plot
uv run flexnet-frames demo/syms/*.sym
uv run flexnet-randomness demo/syms/*.sym
```

The frame report should show 12 constant positions: the sync, the 8 header slots and the
trailer.

## Capture

Any RTL-SDR works. Tune to 901.5 MHz at 2.4 Msps to cover 900.3 to 902.7 MHz, and let it run
for a day:

```bash
rtl_sdr -f 901500000 -s 2400000 - | uv run flexnet-capture 901500000 2400000 bursts/
```

`flexnet-capture` tracks the noise floor in 1 ms blocks and treats anything 10 dB above it as
a burst. For each burst it appends a row to `bursts/bursts.csv` and saves the IQ, with 20 ms
either side, as a complex64 `.cf32` file named after its time, frequency and duration:

```text
20261101T020153.420_901.0623MHz_63ms.cf32
```

It stops saving IQ after 20 GiB but keeps logging; `--max-saved-gb` changes that. A day used
about 11% of one Raspberry Pi 5 core.

**Finding your meter.** Sort `bursts.csv` by `peak_db_over_floor`. Your meter is the strongest
emitter on its channel: ours sat about 40 dB above the floor against 12 to 20 dB for its
neighbours, sending 62 to 63 ms bursts near 901.06 MHz roughly once an hour. A neighbourhood
produces a couple of hundred bursts an hour, so expect most rows to be other meters.

## Demodulate

```bash
uv run flexnet-demod bursts/*901.06*MHz_6[23]ms.cf32 --out-dir syms --plot
```

Each burst becomes a `.sym` file: one line of digit pairs, one pair per symbol, giving the I
then the Q level index 0 to 3 for the levels -3, -1, +1, +3. `--plot` adds a constellation
PNG. The printed score is the RMS distance to the 16-QAM grid in half-spacings. Real frames
score about 0.13 to 0.17, a wrong timing phase about 0.44, and random points about 0.29.

If your capture used a different tuning or rate, pass `--center-hz` and `--rate`. If the burst
frequency isn't in the file name, pass `--freq-hz`.

## Analyse

| Tool | What it does |
| --- | --- |
| `flexnet-frames` | Aligns frames on their trailer and reports constant positions, headers and pairwise differences |
| `flexnet-randomness` | Tests payloads against random data of the same size: chi-square, bit balance, runs, lag correlation, shared runs, entropy |
| `flexnet-fec-search` | Searches for the specification's rate-1/2 K=3 convolutional code under any symbol labelling and many interleavers, by linear algebra over GF(2) |
| `flexnet-idsearch` | Looks for a known number, such as an ID printed on the meter, under 768 symbol-to-bit mappings and common encodings |

Each tool takes `.sym` files and has `--help`. Run them on several frames from one meter.

## How FlexNet Normal mode works

A 2009 Sensus modulation specification is public as an exhibit in the
[FCC's experimental licensing system](https://apps.fcc.gov/els/GetAtt.html?id=109424). It calls
Normal mode "7-FSK, a 16 QAM signal" at 8.04455 kbit/s with 3808 Hz peak deviation, a 28-byte
payload, and an interleaved rate-1/2 K=3 convolutional code.

That means the FM modulating waveform is a subcarrier of about 2011 Hz carrying one 16-QAM
symbol per cycle: 2011 symbols/s × 4 bits = 8044 bit/s, and 3808 / 2011 = 1.89, the
specification's β. The demodulator:

1. mixes the burst to 0 Hz and decimates to 240 kHz;
2. low-passes at 40 kHz, FM-discriminates, and low-passes at 14 kHz;
3. fits the subcarrier on the ~12 ms bare-tone preamble, searching 2000 to 2022 Hz;
4. mixes down by it and rotates the preamble onto the positive I axis;
5. integrates over one subcarrier period per symbol, sweeping 32 timing phases;
6. fits a scale and rotation to the 16-QAM grid and keeps the tightest phase;
7. slices I and Q.

**A trap.** On a discriminator plot the bursts look like 4-level FSK at about 8045 baud, and
cyclostationary analysis finds a clock line there. That line is four times the subcarrier.
Sampled four times a cycle, one 16-QAM symbol appears as I, -Q, -I, Q. No FSK model fits, so
if you see a clean ~2 kHz preamble tone, go straight to the subcarrier demodulator.

The symbol labels are defined up to the 90° rotation the demodulator picks, so compare
symbol strings only between frames decoded by this package.

## The frame

Measured on 14 frames from one meter and checked against neighbouring meters:

| Positions | Field | Observed |
| --- | --- | --- |
| 0-1 | Sync | `03 32`; neighbours share the second symbol, and their first is usually clipped |
| 2, 14, 26 … 86 | Header, 8 symbols (32 bits) | Same in every frame of one meter for 16 hours; different between meters |
| The other 88 | Payload, 352 coded bits | Changes almost completely between frames |
| 98-99 | Trailer | `21 12` on every meter seen |

The header never counted up, so it isn't a sequence number or clock. None of the three numbers
printed on our meter (the radio's FlexNet ID, the utility meter number, and the Sensus serial)
appeared in it or anywhere else in the frame. The 86 ms frames some meters also send use the
same modulation with a different layout: about 148 symbols, with a three-symbol header run
every 18. They aren't mapped yet.

## The payload

Our 14 frames, 1,232 payload symbols, sit inside the 95% range of random data on every
`flexnet-randomness` test. `flexnet-fec-search` finds no convolutional code under any
labelling, any of the interleavers it tries, or a whitening sequence fixed across frames.
Each search was first checked against synthetic frames with a planted answer, and the test
suite repeats those checks.

A March 2025
[Sensus response to Greenville Utilities](https://www.guc.com/sites/default/files/uploads/bids/3-sensus-greenville-response-3-11-2025_final.pdf)
describes AES-256 in CCM mode with CBC-MAC, run on the meter's radio module. It says
encryption is enabled per meter, keyed to the FlexNet ID, optional for the utility, and
"does not significantly increase the size of the packets". That fits a clear address followed
by a structureless body.

Randomness alone can't name a cipher. If your utility doesn't encrypt, though, these tests
should show it: fixed fields, counters, or repeats between frames.

## Open questions

- What the 86 ms frames carry, and whether their header runs are per-meter.
- How the 32-bit header relates to the printed FlexNet ID. Frames from more meters with known
  IDs would help.
- Double Density, Priority and C&I modes, and the GMSK downlink at 930-960 MHz.
- Whether the roughly hourly schedule changes with season, load or utility configuration.

## Development

```bash
uv sync --extra plot
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

The tests run entirely on synthetic data. Please don't add real captures to the repository.

## Sources

- [Sensus FlexNet modulation specification, 2009](https://apps.fcc.gov/els/GetAtt.html?id=109424), FCC experimental licensing exhibit
- [FCC equipment authorization search](https://apps.fcc.gov/oetcf/eas/reports/GenericSearch.cfm), grantee code SDB: the IDTB electricity-meter transceivers
- [Sensus response to Greenville Utilities, March 2025](https://www.guc.com/sites/default/files/uploads/bids/3-sensus-greenville-response-3-11-2025_final.pdf)
- [rtlamr issue #60](https://github.com/bemasher/rtlamr/issues/60): FlexNet is unsupported by rtlamr
- [Home Assistant forum: Sensus Stratus IQ](https://community.home-assistant.io/t/sensus-stratus-iq-smart-meter-just-installed-can-i-get-data-from-it/729971)

## License

MIT. See [LICENSE](LICENSE).
