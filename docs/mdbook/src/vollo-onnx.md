# ONNX Support

VOLLO also provides a tool for compiling ML models defined in ONNX.

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

- Which transformations to perform on the model (see [Example 2: Batched MLP](example-2-batched-mlp.md) for the unweaving transform and [Example 3: CNN](example-3-cnn.md) for the streaming transform):

  ```text
  --streaming-transform <STREAMING_AXIS>
        Axis on which to perform the streaming transform in the NNIR graph

        If unspecified, no streaming transform is performed

  --unweave
        Perform the unweaving transform
  ```

- The input shape of the model. This is required if the ONNX model has dynamic input shapes. VOLLO requires that the shape of the input be known at compile-time:

  ```text
  --override-input-shape <SHAPE>
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
