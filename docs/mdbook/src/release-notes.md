# Release Notes

## 0.17.2

- Fix various issues around DMA buffer handling

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
