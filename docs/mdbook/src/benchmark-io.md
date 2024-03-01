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

| v output \ input > |   32|   64|  128|  256|  512| 1024| 2048| 4096| 8192|
|-------------------:|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|             **32** | 3.6 | 3.6 | 3.7 | 3.7 | 3.9 | 3.9 | 4.3 | 4.8 | 6.1 |
|             **64** | 3.7 | 3.7 | 3.7 | 3.7 | 3.9 | 4.0 | 4.3 | 4.8 | 6.2 |
|            **128** | 3.7 | 3.7 | 3.7 | 3.8 | 3.9 | 4.1 | 4.3 | 4.9 | 6.1 |
|            **256** | 3.7 | 3.7 | 3.8 | 3.8 | 3.9 | 4.0 | 4.4 | 4.9 | 6.1 |
|            **512** | 4.0 | 3.8 | 3.9 | 4.0 | 4.0 | 4.1 | 4.4 | 5.0 | 6.2 |
|           **1024** | 3.9 | 3.9 | 3.9 | 4.0 | 4.1 | 4.2 | 4.5 | 5.1 | 6.3 |
|           **2048** | 4.0 | 4.1 | 4.1 | 4.2 | 4.3 | 4.4 | 4.7 | 5.5 | 6.6 |
|           **4096** | 4.5 | 4.5 | 4.6 | 4.6 | 4.7 | 4.9 | 5.2 | 6.0 | 7.0 |
|           **8192** | 5.6 | 5.5 | 5.6 | 5.7 | 5.8 | 5.9 | 6.3 | 6.8 | 7.9 |
