# Benchmarks

This section provides benchmarks for the Vollo accelerator for a variety of models.

Performance figures are given for the following configurations of the Vollo
accelerator:

- a 6 core, block size 32 configuration which is provided for the V80 accelerator card
- a 3 core, block size 64 configuration which is provided for the IA-840F accelerator
  card
- a 6 core, block size 32 configuration which is provided for the IA-420F accelerator
  card

If you require a different configuration, please contact us at <vollo@myrtle.ai>.

All these performance numbers can be measured using the `vollo-sdk` with the correct accelerator card
by running the provided [benchmark script](running-the-benchmark.md).

We also provide performance figures for a PCIe roundtrip for various input and output sizes.
