# Changelog

## 25.1.1

- Fix bug in `vollo-tool` where `vollo-tool fpga-config` did not enumerate the
  V80 management physical function
- Fix bug in `load-kernel-driver.sh` where the Vollo driver was loaded for the
  V80 management physical function instead of the AMI driver

## 25.1.0

- Early access support for Mamba models
- Add support for SiLU
- Add support for Softplus
- Add support for Exp, Exp2
- Add support for Sigmoid
- Add support for Softmax
- Speed up model compilation

## 25.0.1

- Add support for size 0 input and output tensors in the compiler
- Add `vollo_rt_model_input_shape_len` and `vollo_rt_model_output_shape_len`
  functions to Vollo RT
- Improve constant folding in the compiler

## 25.0.0

- Add early access support for Silicom fb4CGg3@VU09P
- "Spaced" inference latency optimization now applies to multi-model programs
  too
- Increase clock speed on V80 from 280MHz to 300MHz.
- Compiler and scheduler optimizations to improve inference latency
- Speed up model compilation
- Add support for bidirectional LSTMs

## 24.0.1

- Improve latency of streaming models, especially e.g. large LSTMs, when
  inferences have spacing between them, i.e. not being run back-to-back
- Add `spaced` flag to `vollo_compiler.Program.cycle_count_per_inference` to
  get cycle counts of programs with or without spacing in between inferences
- Speed up model compilation

## 24.0.0

- Improved latency when using multiple models
- Improved V80 performance
- Scheduler improvements, especially for models with large layers
- `remember_allocations` argument added to `vollo_compiler.ProgramBuilder` to
  help fit more models in multi-model programs

## 23.1.0

- Add initial support for Alveo V80, further performance optimisations still outstanding
- Add support for Napatech NT400D11
- Add support for vfio-pci; use `load-kernel-driver.sh vfio` to load it,
  required for V80
- Add lock to Vollo RT to prevent concurrent usage of the accelerator
- Improve VM cycle count estimates for Agilex devices
- Additional model support:
  - Add support for broadcasting non-constant tensors except along the data dimension
  - Add grouped convolution support to `vollo_torch.nn.PaddedConv1d`
  - Add support for reshape operations
  - Changes to the API of `vollo_torch.nn.Scan`: the `step` function now returns
    an output tensor and a separate state tensor instead of a single tensor; the
    `forward` method now takes both an `input_axis` and an `output_axis` instead
    of a single `axis` argument

## 22.1.0

- Update compatibility with newer IA420F boards (IA420F-0015)

## 22.0.1

- Allow weights to be shared in multi-model programs

## 22.0.0

- Add support for compiling models with multiple input tensors and multiple
  output tensors
- Improve accuracy of LSTM unit
- Change behavior of VOLLO_FP32_ROUND in Vollo RT so that it's enabled by
  default; set to 0 to truncate f32 inputs
- Change `vollo-tool reset --pci` to `vollo-tool reset-pci`
- Expand supported PyTorch stacking and concatenating operations:
  `concatenate`, `stack`, `vstack`, `hstack`, `row_stack`, `column_stack`,
  `dstack`
- Expand supported PyTorch transposition operations: `permute`,
  `swapdims`, `swapaxes`, `t`, `T`, `mT`

## 21.1.1

- Initial support for `torch.nn.LSTM`
- Performance improvements in VM simulation, especially for LSTMs
- Improve error messages from `vollo_torch.fx.nnir.to_nnir` for unsupported
  field accesses (`getattr`) in PyTorch model
- Add `f32_round` argument to `vollo_compiler.VM.run` methods to choose whether
  to round or truncate f32 inputs (previously always rounded)
- Fix handling of non-contiguous input arrays/tensors in Vollo RT Python
  bindings
- Fix bug in `streaming_transform` for tensor sum reductions

## 21.1.0

- Support tanh

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
