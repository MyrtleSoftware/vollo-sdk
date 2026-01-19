# Example 4: Mixed Precision

By default, operations on Vollo run in `bfloat16 (BF16)` format[^note], but there is support for some operations to be run in float32 (FP32).

See [Supported Models](supported-models.md) for a list of which operations have FP32 support.

To run operations in FP32, the operations should be placed in a `vollo_torch.Fp32Activations` context. The following example
shows an MLP with some pre-processing on its inputs in FP32:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import vollo_torch

class PreprocessMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, y):
        with vollo_torch.Fp32Activations():
            z = x + 0.763 * y
            z = torch.clamp(z, -2.633, 2.633)

        z = F.relu(self.fc1(z))
        residual = z
        z = F.relu(self.fc2(z)) + residual
        return self.out(z)

input_size = 784
output_size = 10
hidden_size = 128
model = PreprocessMLP(input_size, output_size, hidden_size)
```

The inputs and outputs of a model can also be FP32. The default precision is BF16; if FP32 is required this must be
specified when calling `to_nnir`:

```python
import vollo_compiler

inputs = [torch.randn(input_size), torch.randn(input_size)]
(model, expected_output) = vollo_torch.fx.prepare_shape(model, *inputs)
nnir = vollo_torch.fx.nnir.to_nnir(
    model,
    inputs_precisions = [vollo_compiler.NumberFormat.FP32, vollo_compiler.NumberFormat.FP32],
    outputs_precisions = [vollo_compiler.NumberFormat.BF16]
)

# Note that the printed NNIR will be annotated with the precisions of each layer
# (See the activation_precision fields of the layers)
print(nnir)
```

Note that the model's `inputs_precisions` and `outputs_precisions` will determine what type of data format is sent/received between the [Vollo runtime](vollo-runtime.md) and the Vollo accelerator. If possible, it is best to make the precisions in the model match the
precisions of data you will be providing to the runtime. If these precisions do not match, the values will be converted in software
by the runtime, which can be slow.

```python
config = vollo_compiler.Config.ia_420f_c6b32()
program = nnir.to_program(config)
```

[^note]: This is true of most operations, however intermediate values in layer calculations are sometimes stored in higher precision,
e.g. in the accumulation of dot products.
