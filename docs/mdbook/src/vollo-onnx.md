# ONNX Support

Vollo also provides a tool for compiling ML models defined in ONNX.

`vollo-onnx` is a command line tool which allows the user to specify an input ONNX file and produces a `.vollo` program as output. The user specifies a path to the input `.onnx` file:

```text
Arguments:
  <INPUT>
          Path to the input .onnx file
```

The user can specify:

- The output path:

  ```text
  -o, --output <OUTPUT>
        Output path for the compiled program file

        [default: program.vollo]
  ```

- The hardware configuration to use (this JSON file can be generated using the `Config` method `save` in `vollo_compiler`):

  ```text
      --hw-config <HW_CONFIG_JSON>
        Path to the hardware config JSON file

        If unspecified, defaults to the 6 core block size 32 IA_420F configuration
  ```

- Which transformations to perform on the model. Currently the only available transformation is the streaming transform [Example 2: CNN](example-2-cnn.md):

  ```text
  --streaming-transform <STREAMING_AXIS>
        Axis on which to perform the streaming transform in the NNIR graph

        If unspecified, no streaming transform is performed
  ```

- The input shape of the model. This is required if the ONNX model has dynamic input shapes. Vollo requires that the shape of the input be known at compile-time:

  ```text
  --override-input-shape <SHAPE>
        If the model has dynamic input shapes, the user must pass a fixed input shape

        Example: 10,100,250
  ```

## Simplifying ONNX Models

`vollo-onnx` has a limited list of supported ONNX nodes. Often ONNX models can be over-complicated, and contain unnecessary shaping operations. It is recommended that [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) be used before calling `vollo-onnx` on an ONNX model to remove these unnecessary shaping operations which aren't supported by `vollo-onnx`:

```sh
onnx-sim <model.onnx> <model-sim.onnx> --overwrite-input-shape <model-input-shape>
```

It is also recommended to use the `--overwrite-input-shape` with `onnx-simplifier`, as this can enable further simplifications and better constant folding.

## Using ONNX from Python

ONNX models can also be imported and translated to NNIR models directly in python using the static `NNIR` method `from_onnx`. This also requires that the input shape be specified if the ONNX model has dynamic input shapes, otherwise it can be `None`.

```python
onnx_nnir = vollo_compiler.NNIR.from_onnx(onnx_path, input_shape)
```

## Supported Nodes

Tensors are expected to be in `float32` format, unless they are used as indices / axes (in which case they should be `int64`s).

`vollo-onnx` supports models with the following nodes:

| Operator                 | Support Notes                                                        |
| ------------------------ | -------------------------------------------------------------------- |
| Pointwise arithmetic ops | `Add`, `Sub`, `Mul`; `Div` by constant                               |
| Inequality               | `>`, `<`, `>=`, `<=` (when followed by a `Where`)                    |
| `Max` and `Min`          |                                                                      |
| `Neg`                    |                                                                      |
| Clamp ops                | `Clip`, `Relu`                                                       |
| Matrix multiplication    | `MatMul` / `Gemm` where one input is a constant                      |
| `Conv`                   | 1d with left-padding such that input and output seq dimensions match |
| `LSTM`                   | Forward LSTM without explicit hidden or cell state initialisation    |
| `Gather`                 | With a 1d tensor of indices                                          |
| `ReduceSum`              | With constant axes                                                   |
| `Where`                  | If the `Where` condition is an inequality comparison                 |
| `Concat`                 | On outer dimension or at start or end of model                       |
| `Transpose`              | See [tensor memory format](supported-models.md#tensor-memory-format) |
| `Squeeze`, `Unsqueeze`   |                                                                      |
| `Identity`               |                                                                      |
