"""Receive-only tools for Sensus FlexNet electric-meter transmissions.

The package covers the whole path from an RTL-SDR to analysed symbols:

* :mod:`flexnet_rx.capture` records every radio burst from an ``rtl_sdr`` IQ stream.
* :mod:`flexnet_rx.demod` turns one recorded burst into hard-decided 16-QAM symbols.
* :mod:`flexnet_rx.frame` splits frames into sync, header, payload and trailer.
* :mod:`flexnet_rx.randomness` tests whether a payload is distinguishable from random.
* :mod:`flexnet_rx.fec` searches for the convolutional code the specification describes.
* :mod:`flexnet_rx.idsearch` looks for a known number (such as a printed meter ID).
* :mod:`flexnet_rx.synth` generates synthetic bursts and coded frames for testing.

Nothing here transmits, and nothing attempts to decrypt anything.
"""

__version__ = "0.1.0"
