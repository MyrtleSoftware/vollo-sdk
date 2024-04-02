# IO Round trip

The following IO round trip times are sampled by using a program with no compute on the
Vollo accelerator from the Vollo runtime.

More specifically this Vollo accelerator program waits for the last input byte to arrive
before it sends the first output byte back. This method takes into account some of the
overheads (such as copying to the DMA buffer in the Vollo runtime) associated with IO
and this test is set up to see how it scales with difference sizes of inputs and output
values.

**NOTE: We are currently working on an improved DMA solution which will significantly reduce these times**

The following table shows the round trip times in `μs` on the `IA840F`
board (similar times were observed on `IA420F`), each value is a `bfloat16` (2 bytes),
using fewer than 32 values gives the same times as 32 values:

| out\in |   32|   64|  128|  256|  512| 1024| 2048| 4096| 8192|
|-------:|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|  **32**| 3.3 | 3.3 | 3.4 | 3.4 | 3.5 | 3.7 | 4.0 | 4.6 | 5.8 |
|  **64**| 3.3 | 3.3 | 3.4 | 3.4 | 3.5 | 3.7 | 4.0 | 4.6 | 5.8 |
| **128**| 3.4 | 3.4 | 3.4 | 3.5 | 3.6 | 3.7 | 4.0 | 4.6 | 5.8 |
| **256**| 3.4 | 3.4 | 3.5 | 3.5 | 3.6 | 3.7 | 4.0 | 4.6 | 5.8 |
| **512**| 3.5 | 3.5 | 3.5 | 3.6 | 3.7 | 3.9 | 4.1 | 4.7 | 5.9 |
|**1024**| 3.6 | 3.5 | 3.6 | 3.6 | 3.9 | 3.9 | 4.2 | 4.8 | 6.0 |
|**2048**| 3.8 | 3.8 | 3.9 | 3.9 | 3.9 | 4.1 | 4.4 | 5.0 | 6.2 |
|**4096**| 4.2 | 4.2 | 4.3 | 4.3 | 4.4 | 4.6 | 4.9 | 5.5 | 6.7 |
|**8192**| 5.3 | 5.2 | 5.3 | 5.3 | 5.5 | 5.6 | 5.9 | 6.6 | 7.9 |

To reproduce these values on your own hardware run the provided [benchmark
script](running-the-benchmark.md) with environment variable `RUN_IO_TEST=1`.
