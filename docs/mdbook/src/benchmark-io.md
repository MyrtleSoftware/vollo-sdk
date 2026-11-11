# IO Round Trip

The following IO round trip times are sampled by using a program with no compute on the
Vollo accelerator from the Vollo runtime.

More specifically this Vollo accelerator program waits for the last input byte to arrive
before it sends the first output byte back. This method takes into account some of the
overheads (such as copying between the application's buffers and the runtime's DMA
buffers) associated with IO and this test is set up to see how it scales with
difference sizes of inputs and outputs.

The following tables show the round trip times in `μs` on the V80 board, for different
input and output sizes, with the runtime's defaults. Using fewer than 64 bytes gives the
same times as 64 bytes.

To reproduce these values on your own hardware run the provided [benchmark
script](running-the-benchmark.md) with environment variable `RUN_IO_TEST=1`.

By default the runtime currently sends an inference's input over MMIO, written straight
from the application's buffer, when the model's inputs total at most 8 KiB, and over DMA
otherwise. MMIO also needs no DMA input in flight and room in the card's input ring, so
with several inferences in flight a small input can still go over DMA. These tables run
one inference at a time, so every input up to 8192 B goes over MMIO and only the 16384 B
column goes over DMA. The `VOLLO_MMIO_MAX_SIZE` environment variable sets the threshold
in bytes, and `0` sends every input over DMA. Outputs always come back over DMA.

## User buffers

The application passes its own buffers. The runtime copies each output from its DMA
buffer into the application's buffer. An input is only copied into a DMA buffer when it
goes over DMA (the 16384 B column); up to 8 KiB it is written over MMIO, with no copy.

**mean**:

|     out\\in |   64 B |   128 B |   256 B |   512 B |   1024 B |   2048 B |   4096 B |   8192 B |   16384 B |
|------------:|-------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|----------:|
|    **64 B** |   0.86 |    0.87 |    0.87 |    0.89 |     0.94 |     1.02 |     1.21 |     1.56 |      2.82 |
|   **128 B** |   0.92 |    0.92 |    0.93 |    0.95 |     0.99 |     1.08 |     1.26 |     1.62 |      2.90 |
|   **256 B** |   0.96 |    0.97 |    0.95 |    0.99 |     1.03 |     1.12 |     1.28 |     1.65 |      2.91 |
|   **512 B** |   1.00 |    1.02 |    1.03 |    1.03 |     1.10 |     1.19 |     1.37 |     1.71 |      2.96 |
|  **1024 B** |   1.07 |    1.05 |    1.06 |    1.10 |     1.15 |     1.24 |     1.41 |     1.77 |      3.03 |
|  **2048 B** |   1.14 |    1.14 |    1.18 |    1.21 |     1.23 |     1.30 |     1.46 |     1.89 |      3.10 |
|  **4096 B** |   1.38 |    1.36 |    1.35 |    1.38 |     1.44 |     1.51 |     1.74 |     2.05 |      3.35 |
|  **8192 B** |   1.76 |    1.90 |    1.80 |    1.82 |     1.88 |     1.93 |     2.15 |     2.51 |      3.88 |
| **16384 B** |   2.71 |    2.72 |    2.71 |    2.71 |     2.79 |     2.94 |     3.05 |     3.46 |      4.72 |

**p99**:

|     out\\in |   64 B |   128 B |   256 B |   512 B |   1024 B |   2048 B |   4096 B |   8192 B |   16384 B |
|------------:|-------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|----------:|
|    **64 B** |   0.88 |    0.88 |    0.89 |    0.91 |     0.97 |     1.05 |     1.23 |     1.59 |      3.04 |
|   **128 B** |   1.23 |    1.23 |    1.24 |    1.26 |     1.31 |     1.39 |     1.57 |     1.93 |      3.15 |
|   **256 B** |   1.26 |    1.26 |    1.27 |    1.30 |     1.34 |     1.43 |     1.60 |     1.96 |      3.15 |
|   **512 B** |   1.31 |    1.32 |    1.32 |    1.33 |     1.34 |     1.49 |     1.65 |     2.00 |      3.27 |
|  **1024 B** |   1.35 |    1.34 |    1.35 |    1.36 |     1.43 |     1.51 |     1.68 |     2.06 |      3.32 |
|  **2048 B** |   1.43 |    1.43 |    1.44 |    1.47 |     1.51 |     1.59 |     1.76 |     2.12 |      3.38 |
|  **4096 B** |   1.66 |    1.66 |    1.67 |    1.69 |     1.74 |     1.81 |     2.01 |     2.35 |      3.63 |
|  **8192 B** |   2.04 |    2.35 |    2.06 |    2.07 |     2.13 |     2.44 |     2.60 |     2.77 |      4.07 |
| **16384 B** |   3.16 |    3.16 |    3.15 |    3.15 |     3.22 |     3.36 |     3.50 |     3.90 |      5.41 |

## Raw DMA buffers

This is using buffers allocated with `vollo_rt_get_raw_buffer`, which lets the runtime
skip the copies to and from its DMA buffers.

**mean**:

|     out\\in |   64 B |   128 B |   256 B |   512 B |   1024 B |   2048 B |   4096 B |   8192 B |   16384 B |
|------------:|-------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|----------:|
|    **64 B** |   0.85 |    0.84 |    0.85 |    0.87 |     0.92 |     1.01 |     1.19 |     1.54 |      2.77 |
|   **128 B** |   0.85 |    0.86 |    0.86 |    0.88 |     0.93 |     1.02 |     1.20 |     1.55 |      2.77 |
|   **256 B** |   0.88 |    0.88 |    0.88 |    0.90 |     0.94 |     1.04 |     1.22 |     1.57 |      2.80 |
|   **512 B** |   0.91 |    0.91 |    0.92 |    0.94 |     0.99 |     1.07 |     1.25 |     1.61 |      2.84 |
|  **1024 B** |   0.94 |    0.94 |    0.95 |    0.97 |     1.02 |     1.10 |     1.27 |     1.64 |      2.91 |
|  **2048 B** |   1.01 |    1.01 |    1.02 |    1.05 |     1.09 |     1.18 |     1.34 |     1.70 |      2.91 |
|  **4096 B** |   1.15 |    1.15 |    1.16 |    1.18 |     1.22 |     1.31 |     1.49 |     1.85 |      3.07 |
|  **8192 B** |   1.42 |    1.42 |    1.43 |    1.45 |     1.49 |     1.58 |     1.77 |     2.12 |      3.34 |
| **16384 B** |   1.97 |    1.97 |    1.98 |    2.01 |     2.04 |     2.14 |     2.32 |     2.68 |      3.89 |

**p99**:

|     out\\in |   64 B |   128 B |   256 B |   512 B |   1024 B |   2048 B |   4096 B |   8192 B |   16384 B |
|------------:|-------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|----------:|
|    **64 B** |   0.86 |    0.86 |    0.87 |    0.89 |     0.94 |     1.04 |     1.21 |     1.57 |      2.96 |
|   **128 B** |   0.87 |    0.87 |    0.87 |    0.89 |     0.95 |     1.04 |     1.22 |     1.57 |      2.94 |
|   **256 B** |   0.89 |    0.89 |    0.90 |    0.92 |     0.97 |     1.06 |     1.24 |     1.60 |      3.10 |
|   **512 B** |   0.92 |    0.93 |    0.94 |    0.95 |     1.02 |     1.10 |     1.27 |     1.63 |      3.17 |
|  **1024 B** |   0.96 |    0.96 |    0.97 |    0.99 |     1.04 |     1.12 |     1.29 |     1.66 |      3.09 |
|  **2048 B** |   1.02 |    1.03 |    1.04 |    1.06 |     1.11 |     1.20 |     1.37 |     1.73 |      2.96 |
|  **4096 B** |   1.16 |    1.17 |    1.18 |    1.20 |     1.23 |     1.33 |     1.51 |     1.87 |      3.36 |
|  **8192 B** |   1.44 |    1.43 |    1.44 |    1.46 |     1.51 |     1.60 |     1.79 |     2.15 |      3.60 |
| **16384 B** |   1.99 |    1.99 |    2.00 |    2.03 |     2.06 |     2.16 |     2.35 |     2.71 |      3.91 |
