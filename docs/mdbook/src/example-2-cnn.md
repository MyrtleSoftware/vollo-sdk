# Example 2: CNN

Vollo supports streaming 1D convolutional neural networks (CNNs), which might
require you to make some changes to your model if you are currently using a
non-streaming 1D CNN.

A streaming convolution applies the convolutional kernel to the most recent
window of the input sequence as the data points in the input sequence arrive.
This differs from a non-streaming convolution, which expects to receive a
complete input sequence and applies its convolutional kernel to each window of
that input.

Streaming convolutions will have much lower latency than non-streaming
convolutions, but they have to maintain some state, namely the most recent
window of input, making them unnatural to define in ML frameworks like PyTorch.
To enable the use of of streaming convolutions, the Vollo compiler includes a
`streaming_transform` which transforms a non-streaming CNN into a streaming CNN,
as long as the non-streaming CNN meets certain constraints.

## Using the `streaming_transform`

The model below is a non-streaming CNN taking an input sequence of length 5 and
producing an output of length 1.
(It can actually take any input sequence of length 5+n and produce an output of
length 1+n, but we will only consider the minimal sequence length, since that is
the length of the input context used by each of the output elements.)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size=3):
        super().__init__()
        # Reduces sequence length by (kernel_size - 1) = 2
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size)
        # Reduces sequence length by (kernel_size - 1) = 2
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# Instantiate the model
in_channels = 32
out_channels = 1
hidden_channels = 128
model = CNN(in_channels, out_channels, hidden_channels)
```

In order to apply the `streaming_transform`, the `torch.nn.Conv1d` layers need
to be replaced with `vollo_torch.nn.PaddedConv1d` layers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

import vollo_torch.nn

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.conv1 = vollo_torch.nn.PaddedConv1d(in_channels, hidden_channels, kernel_size)
        self.conv2 = vollo_torch.nn.PaddedConv1d(hidden_channels, out_channels, kernel_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# Instantiate the model
in_channels = 32
out_channels = 1
hidden_channels = 128
model = CNN(in_channels, out_channels, hidden_channels)
```

These `PaddedConv1d` layers are identical to `torch.nn.Conv1d`, but with left
padding pre-applied to the input so as not to reduce the sequence length.

This `PaddedConv1d` model is still a non-streaming model, which now takes an
input sequence of length 5 and produces an output of length 5.
Its relationship to the original `Conv1d` model is that, given the same model
parameters (weights, biases, etc.) and input sequence, the last element of the
output sequence of the `PaddedConv1d` model will be equal to the last/only
element of the output sequence of the `Conv1d` model.

The `PaddedConv1d` model can be lowered to NNIR and have the
`streaming_transform` applied.

```python
batch_size = 1
sequence_length = 5
input = torch.randn(batch_size, in_channels, sequence_length)
(model, expected_output) = vollo_torch.fx.prepare_shape(model, input)
nnir = vollo_torch.fx.nnir.to_nnir(model)

# Provide the streaming transform with index of the sequence axis
(nnir, output_axis) = nnir.streaming_transform(2)
```

The resulting NNIR graph represents a streaming CNN, i.e. containing state, that
takes a single data point of a sequence as input and produces a single data
point as output, updating its input window state in the process.
Input sequences for the streaming CNN need to be fed in sequentially, e.g. in a
loop.
For example, using the VM:

```python
import vollo_compiler

program = nnir.to_program(vollo_compiler.Config.ia_420f_c6b32())
vm = program.to_vm()

vm_outputs = []
for i in range(5):
    # Runs inference on one element of the input sequence, updating the
    # streaming CNN's state
    vm_outputs.append(vm.run(input[:, :, i].detach().numpy()))

torch.testing.assert_close(
    expected_output,
    torch.stack(
        [torch.from_numpy(output) for output in vm_outputs],
        axis=output_axis,
    ),
    atol = 5e-3,
    rtol = 1e-3
)
```

The streaming CNN satisfies the property that, given an input sequence, the i-th
element of the output sequence of the non-streaming CNN will be equal to the
output of the i-th iteration of feeding the input to the streaming CNN.

The streaming CNN can be saved and run on the accelerator like any other
program:

```python
program.save('cnn.vollo')
```
