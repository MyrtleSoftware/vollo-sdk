# Release Notes

## 21.0.0

- Runtime/bitstream optimisation for small inputs (using MMIO instead of DMA)
- Scheduling and architecture optimisations
- Add `reset` subcommand to `vollo-tool`
- Support ReLU via `torch.relu`

## 20.0.3

- Separate bitstreams from Vollo SDK
- Add c2b64d hw config to support models up to 8M parameters (bitstream and compiler)
- Improve compiler error messages
- Fix example build

## 20.0.2

- Fix for incorrect `vollo_rt_accelerator_num_cores` introduced in 20.0.1

## 20.0.1

- `vollo_rt_add_vm` to test the `vollo-rt` API without an accelerator
- `vollo_rt_load_program_from_buffer` and `vollo_compiler.Program.{save,load}_bytes`
- Add `vollo_torch.nn.RecurrentStateLSTM` for modelling streaming LSTM models across forward passes
- Codegen fix for `vollo_torch.nn.Scan`
- Fix incorrect input validation for `torch.sum` layers
- Change vollo-rt example to compile with older C compilers

## 20.0.0

- Add support for LayerNorm
- Add support for RMSNorm
- Add support for sqrt operations (`torch.sqrt` and `torch.rsqrt`)
- Add support for summing over the data dimension
- Add `cycle_count_per_inference` and `compute_duration_per_inference_us` Program methods
- Add support for a wider range of torch arithmetic operation aliases

## 19.2.3

- Downgrade glibc dependency to support systems with glibc >=2.17

## 19.2.2

- Add support for `torch.div`, `torch.Tensor.div`
- Fix compiler code generation bug for division

## 19.2.1

- Add support for scalars on the left of division
- Add support for `Reciprocal` node in ONNX frontend

## 19.2.0

- Add support for division by non-constant tensors
- Fix slicing in ONNX frontend

## 19.1.1

- Fix compiler bug in constant folding

## 19.1.0

- Add support for partial updates of input data on the accelerator
- VM simulates Vollo accelerator bit-accurately: `bf16_precision` argument
  renamed to `bit_accurate` and enabled by default
- `vollo-tool` includes license self-service
- Performance improvements due to DMA optimization

## 18.0.2

- Add `optimize_transforms` option to the compiler to improve program schedule in some cases

## 18.0.1

- Add fallback to Vollo RT and vollo-tool for when AVX is not available

## 18.0.0

- Vollo RT support for using raw DMA buffers to skip IO copy
- Vollo RT remove redundant/noisy warnings on error: it is the user's responsibility to check returned errors
- Compiler optimization for Where nodes
- Compiler scheduling optimizations
- Vollo IP Core public documentation

## 0.17.1

- Fix vollo-tool compatibility with older bitstreams

## 0.17.0

- New DMA engine that reduces IO latencies by ~1.3us
- Initial support for non-streaming LSTM

## 0.16.0

- Vollo IP Core now available on request
- Add C library for configuring IP Core: `vollo-cfg`
- Support for slicing/concatenation in the middle of models
- Support for BatchNorm nodes
- Support for Scan/LSTMCell nodes
- Add `--io-only` option to `vollo-onnx`
- Add `program-metadata` command to `vollo-tool`
- Fix compiler bug with transposing streaming dimension
- Fix accelerator bug in initial state of streaming models

## 0.15.0

- Accelerator bug fix

## 0.14.0

- Support for filtering dropout layers
- Instruction packing improvements
- LSTM performance improvement
- Improvements to weight sharing

## 0.13.0

- Support for multi-model programs
- Provide Python bindings to Vollo RT: `vollo_rt`
- Improved support and error messages for tensor indexing in compiler
- The unweave transform is now automatic

## 0.12.2

- Support for LSTM nodes in ONNX frontend
- Support for squeezing, unsqueezing, reduce sum, using `unweave`
  transformation
- Improved error reporting in `vollo_torch` lowering to NNIR

## 0.12.1

- `vollo-torch` fix type hints being incompatible with Python 3.7/3.8
- `vollo-rt.h` fix namespacing issue (`error_t` -> `vollo_rt_error_t`)
- Runtime optimisations
- Added IO only benchmarks

## 0.12.0

- Initial support for ONNX models in compiler
- Support for LSTM nodes
- Improved error reporting in compiler
- Compiler API changes
- New runtime API with access to model metadata
- HW optimisations (pointwise operations)
- IA840F support

## 0.10.1

- Support for scalar (`int`, `float`) literals in pointwise operations in
  `vollo-torch`.

## 0.10.0

- Architectural changes in bitstream to support compiler
- Reduced latency from reduced core to core communication in the bitstream
- Add general model compiler and VM simulation with Python bindings in
  `vollo-python`
- Add PyTorch frontend to model compiler in `vollo-torch`
