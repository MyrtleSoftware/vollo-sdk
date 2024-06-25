# IO Round trip

The following IO round trip times are sampled by using a program with no compute on the
Vollo accelerator from the Vollo runtime.

More specifically this Vollo accelerator program waits for the last input byte to arrive
before it sends the first output byte back. This method takes into account some of the
overheads (such as copying to the DMA buffer in the Vollo runtime) associated with IO
and this test is set up to see how it scales with difference sizes of inputs and output
values.

The following table shows the round trip times in `μs` on the `IA420F`
board (similar times were observed on `IA840F`), each value is a `bfloat16` (2 bytes),
using fewer than 32 values gives the same times as 32 values:

| out\in |   32|   64|  128|  256|  512| 1024| 2048| 4096| 8192|
|-------:|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|  **32**| 2.0 | 2.0 | 2.1 | 2.0 | 2.1 | 2.3 | 2.6 | 3.2 | 4.5 |
|  **64**| 2.0 | 2.0 | 2.0 | 2.1 | 2.2 | 2.4 | 2.6 | 3.2 | 4.5 |
| **128**| 2.0 | 2.1 | 2.1 | 2.2 | 2.2 | 2.4 | 2.7 | 3.3 | 4.6 |
| **256**| 2.1 | 2.1 | 2.2 | 2.2 | 2.3 | 2.4 | 2.7 | 3.3 | 4.7 |
| **512**| 2.1 | 2.1 | 2.2 | 2.3 | 2.4 | 2.5 | 2.8 | 3.4 | 4.7 |
|**1024**| 2.3 | 2.3 | 2.4 | 2.4 | 2.5 | 2.6 | 2.9 | 3.5 | 4.9 |
|**2048**| 2.4 | 2.5 | 2.6 | 2.6 | 2.7 | 2.9 | 3.2 | 3.7 | 5.1 |
|**4096**| 2.9 | 3.0 | 3.0 | 3.0 | 3.2 | 3.3 | 3.6 | 4.2 | 5.5 |
|**8192**| 4.0 | 4.0 | 4.0 | 4.0 | 4.2 | 4.4 | 4.7 | 5.3 | 6.5 |

To reproduce these values on your own hardware run the provided [benchmark
script](running-the-benchmark.md) with environment variable `RUN_IO_TEST=1`.
