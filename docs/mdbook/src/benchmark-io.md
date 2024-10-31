# IO Round Trip

The following IO round trip times are sampled by using a program with no compute on the
Vollo accelerator from the Vollo runtime.

More specifically this Vollo accelerator program waits for the last input byte to arrive
before it sends the first output byte back. This method takes into account some of the
overheads (such as copying to the DMA buffer in the Vollo runtime) associated with IO
and this test is set up to see how it scales with difference sizes of inputs and output
values.

The following tables shows the round trip times in `μs` on the `IA420F`
board (similar times were observed on `IA840F`), each value is a `bfloat16` (2 bytes),
using fewer than 32 values gives the same times as 32 values.

To reproduce these values on your own hardware run the provided [benchmark
script](running-the-benchmark.md) with environment variable `RUN_IO_TEST=1`.

## User buffers

**mean**:

| out\\in  |   32 |   64 |   128 |   256 |   512 |   1024 |   2048 |   4096 |   8192 |
|---------:|-----:|-----:|------:|------:|------:|-------:|-------:|-------:|-------:|
|   **32** |  1.9 |  1.8 |   1.9 |   2.0 |   1.9 |    2.2 |    2.7 |    3.3 |    4.7 |
|   **64** |  1.9 |  1.9 |   1.9 |   2.0 |   2.0 |    2.2 |    2.8 |    3.4 |    4.7 |
|  **128** |  1.9 |  2.0 |   1.9 |   2.0 |   2.0 |    2.2 |    2.8 |    3.6 |    4.9 |
|  **256** |  2.0 |  2.0 |   1.9 |   2.0 |   2.0 |    2.2 |    2.7 |    3.5 |    4.9 |
|  **512** |  2.0 |  2.0 |   2.0 |   2.0 |   2.0 |    2.2 |    2.7 |    3.5 |    4.8 |
| **1024** |  2.2 |  2.2 |   2.1 |   2.3 |   2.2 |    2.3 |    2.7 |    3.5 |    4.9 |
| **2048** |  2.4 |  2.4 |   2.5 |   2.5 |   2.5 |    2.6 |    2.8 |    3.9 |    5.1 |
| **4096** |  2.9 |  2.9 |   2.9 |   2.9 |   3.0 |    3.2 |    3.7 |    3.6 |    4.9 |
| **8192** |  3.9 |  3.9 |   3.9 |   3.9 |   3.9 |    4.1 |    4.7 |    4.5 |    5.1 |

**p99**:

| out\\in  |   32 |   64 |   128 |   256 |   512 |   1024 |   2048 |   4096 |   8192 |
|---------:|-----:|-----:|------:|------:|------:|-------:|-------:|-------:|-------:|
|   **32** |  2.0 |  2.0 |   2.1 |   2.1 |   2.1 |    2.4 |    2.9 |    3.6 |    4.9 |
|   **64** |  2.1 |  2.1 |   2.1 |   2.1 |   2.2 |    2.4 |    3.0 |    3.7 |    5.0 |
|  **128** |  2.0 |  2.1 |   2.1 |   2.1 |   2.2 |    2.4 |    3.0 |    3.8 |    5.4 |
|  **256** |  2.1 |  2.1 |   2.0 |   2.1 |   2.2 |    2.4 |    3.0 |    3.7 |    5.2 |
|  **512** |  2.1 |  2.1 |   2.2 |   2.2 |   2.2 |    2.4 |    2.9 |    3.8 |    5.2 |
| **1024** |  2.4 |  2.4 |   2.4 |   2.5 |   2.4 |    2.5 |    3.0 |    3.8 |    5.2 |
| **2048** |  2.5 |  2.6 |   2.6 |   2.7 |   2.7 |    2.8 |    3.0 |    4.3 |    5.5 |
| **4096** |  3.2 |  3.2 |   3.2 |   3.2 |   3.3 |    3.5 |    3.9 |    3.9 |    5.2 |
| **8192** |  4.2 |  4.2 |   4.1 |   4.2 |   4.2 |    4.4 |    5.0 |    4.7 |    5.3 |

## Raw DMA buffers

This is using buffers allocated with `vollo_rt_get_raw_buffer` which lets the runtime skip IO copy.

**mean**:

| out\\in  |   32 |   64 |   128 |   256 |   512 |   1024 |   2048 |   4096 |   8192 |
|---------:|-----:|-----:|------:|------:|------:|-------:|-------:|-------:|-------:|
|   **32** |  1.8 |  1.8 |   1.9 |   1.8 |   1.8 |    1.8 |    2.0 |    2.2 |    2.7 |
|   **64** |  1.8 |  1.8 |   1.8 |   1.9 |   1.9 |    1.9 |    2.0 |    2.2 |    2.7 |
|  **128** |  1.8 |  1.8 |   1.8 |   1.9 |   1.9 |    1.9 |    2.0 |    2.2 |    2.6 |
|  **256** |  1.8 |  1.8 |   1.8 |   1.8 |   1.8 |    1.9 |    2.0 |    2.2 |    2.7 |
|  **512** |  1.8 |  1.8 |   1.9 |   1.8 |   1.9 |    1.9 |    2.1 |    2.3 |    2.7 |
| **1024** |  1.9 |  1.9 |   1.9 |   1.9 |   1.9 |    2.0 |    2.1 |    2.3 |    2.7 |
| **2048** |  1.9 |  2.0 |   2.0 |   2.0 |   2.0 |    2.1 |    2.2 |    2.4 |    2.8 |
| **4096** |  2.2 |  2.2 |   2.3 |   2.2 |   2.2 |    2.3 |    2.4 |    2.6 |    3.0 |
| **8192** |  2.5 |  2.6 |   2.6 |   2.6 |   2.6 |    2.7 |    2.8 |    3.0 |    3.4 |

**p99**:

| out\\in  |   32 |   64 |   128 |   256 |   512 |   1024 |   2048 |   4096 |   8192 |
|---------:|-----:|-----:|------:|------:|------:|-------:|-------:|-------:|-------:|
|   **32** |  1.9 |  1.9 |   2.0 |   1.9 |   2.0 |    2.0 |    2.2 |    2.4 |    3.0 |
|   **64** |  1.9 |  1.9 |   2.0 |   1.9 |   2.0 |    2.1 |    2.2 |    2.5 |    3.0 |
|  **128** |  1.9 |  1.9 |   2.0 |   2.0 |   2.1 |    2.1 |    2.2 |    2.5 |    2.9 |
|  **256** |  1.9 |  1.9 |   1.9 |   2.0 |   2.0 |    2.1 |    2.2 |    2.4 |    3.0 |
|  **512** |  2.0 |  1.9 |   2.0 |   2.0 |   2.0 |    2.1 |    2.2 |    2.5 |    3.0 |
| **1024** |  2.1 |  2.0 |   2.0 |   2.1 |   2.1 |    2.2 |    2.3 |    2.5 |    3.0 |
| **2048** |  2.1 |  2.1 |   2.2 |   2.2 |   2.2 |    2.2 |    2.4 |    2.6 |    3.0 |
| **4096** |  2.3 |  2.3 |   2.4 |   2.4 |   2.4 |    2.4 |    2.6 |    2.9 |    3.4 |
| **8192** |  2.8 |  2.7 |   2.8 |   2.8 |   2.8 |    2.9 |    3.0 |    3.2 |    3.7 |