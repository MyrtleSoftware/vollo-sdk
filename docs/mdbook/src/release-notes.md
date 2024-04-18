# Release Notes

## 0.13.1

- Fixed a bug in long running Vollo RT

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
