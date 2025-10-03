# Example 3: LSTM

Vollo supports both streaming and non-streaming LSTM models.

## Non-streaming LSTM model

To compile an LSTM model as a non-streaming model, you can follow the steps outlined in [Example 1: MLP](example-1-mlp.md) of lowering the model to NNIR, and then compiling it with an accelerator config.

```python
import torch
import torch.nn as nn
import vollo_compiler
import vollo_torch

class LstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

input_size = 128
hidden_size = 256
num_layers = 2
output_size = 16
model = LstmNet(input_size, hidden_size, num_layers, output_size)

seq_length = 20

input = torch.randn(seq_length, input_size)
# Trace the model's execution to annotate it with activation shapes
(model, expected_output) = vollo_torch.fx.prepare_shape(model, input)
nnir = vollo_torch.fx.nnir.to_nnir(model)

# Replace this config with the one for the accelerator you are using
config = vollo_compiler.Config.ia_840f_c3b64()
program = nnir.to_program(config)
```

The example above will give a Vollo program which takes an input tensor of size `20 x 128`,
and will output a tensor of size `20 x 16`. This is a non-streaming LSTM model, since it takes
an entire sequence as input.

As in [Example 1: MLP](example-1-mlp.md) we can construct a VM instance to simulate the Vollo accelerator,
allowing us to get bit-accurate results from the compiled model, and a latency estimate.

```python
vm = program.to_vm()
vm_output = vm.run(input.detach().numpy())
torch.testing.assert_close(expected_output, torch.from_numpy(vm_output), atol = 1e-2, rtol = 1e-2)
print(f"latency (compute): {program.compute_duration_per_inference_us():.1f}us")
```

## Streaming LSTM model

If you want the LSTM model to operate on an ongoing stream of data as its input sequence, it is probably
more desirable to use a streaming LSTM model. We can use the same PyTorch model defined above. The only additional
step required is to call the `streaming_transform` (detailed in [`Example 2: CNN`](example-2-cnn.md)) on the NNIR:

```python
input_size = 128
hidden_size = 256
num_layers = 2
output_size = 16
model = LstmNet(input_size, hidden_size, num_layers, output_size)

seq_length = 20

input = torch.randn(seq_length, input_size)
# Trace the model's execution to annotate it with activation shapes
(model, expected_output) = vollo_torch.fx.prepare_shape(model, input)
nnir = vollo_torch.fx.nnir.to_nnir(model)
# We provide the streaming transform with the sequence axis to 'stream' over.
(nnir, output_streaming_axis) = nnir.streaming_transform(0)
assert(output_streaming_axis == 0)
```

Here, the `streaming_transform` tells the compiler to treat axis 0 as the sequence
dimension, and that we intend to provide the program with a single sequence element
per inference. I.e. on each inference, we pass in a tensor of size `128` and receive a
tensor of size `16` on each inference. The resulting Vollo program is stateful, and will
update the internal hidden state and cell state on each inference.

Note that this streaming model will have a much lower latency on Vollo than the non-streaming model.
The streaming model only needs to run `num_layers = 2` LSTM operations per inference, where the non-streaming
model needs to run `num_layers * seq_length = 2 * 20` LSTM operations per inference.

As above, we can now compile this streaming NNIR to a program with a chosen accelerator
configuration, and test the program with a VM:

```python
# Replace the Config in the line below with the Config for the accelerator you
# are using
program = nnir.to_program(vollo_compiler.Config.ia_840f_c3b64())
vm = program.to_vm()

vm_outputs = []
for i in range(seq_length):
    # Note that the VM takes a single sequence element per run
    vm_outputs.append(vm.run(input[i, :].detach().numpy()))

torch.testing.assert_close(
    expected_output,
    torch.stack(
        [torch.from_numpy(output) for output in vm_outputs],
        axis=output_streaming_axis,
    ),
    atol = 1e-2,
    rtol = 1e-2
)
print(f"latency (compute): {program.compute_duration_per_inference_us():.1f}us")
```
