# Example 2: Batched MLP

In this example, we will use the same `MLP` model as in [Example
1](example-1-mlp.md), but with a batch of inputs instead of a single input
vector.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        residual = x
        x = F.relu(self.fc2(x)) + residual
        return self.out(x)

# Instantiate the model
input_size = 784
output_size = 10
hidden_size = 128
model = MLP(input_size, output_size, hidden_size)

import vollo_torch

# Create a batch of inputs
batch_size = 8
input = torch.randn(batch_size, input_size)
(model, expected_output) = vollo_torch.fx.prepare_shape(model, input)
nnir = vollo_torch.fx.nnir.to_nnir(model)
```

The Vollo accelerator is a vector (1D) processor and the Vollo compiler
optimises for operating on vectors, so the NNIR graph needs to be transformed
to "unbatch" the 2D batch of inputs before it can be compiled to a program.

The unweaving transform can be used to take a model that operates on a batch of
vectors and turn it into a model that operates on each of the vectors
independently.

```python
nnir = nnir.unweave()
```

The model can now be compiled and simulated as in Example 1.

```python
import vollo_compiler

config = vollo_compiler.Config.ia_420f_c6b32()
program = nnir.to_program(config)
program.save('mlp.vollo')
```

## Other Uses of the `unweave` Transform

The unweaving transform is more general than turning batched computations into
unbatched computations.
A model can also be unweaved if it accepts multiple input vectors, but with
_different_ code paths for each of them.
For example, the model below applies a different linear layer to each of the two
input vectors that are contained in the input `xy`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, xy):
        # xy has shape [2, input_size]
        x = torch.index_select(xy, 0, torch.tensor([0]))
        y = torch.index_select(xy, 0, torch.tensor([1]))
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        return self.out(x + y)

# Instantiate the model
input_size = 32
output_size = 1
hidden_size = 128
model = MultiMLP(input_size, output_size, hidden_size)

import vollo_compiler
import vollo_torch

input = torch.randn(2, input_size)
(model, expected_output) = vollo_torch.fx.prepare_shape(model, input)
nnir = vollo_torch.fx.nnir.to_nnir(model)
nnir = nnir.unweave()
config = vollo_compiler.Config.ia_420f_c6b32()
program = nnir.to_program(config)

# Test the program
vm = program.to_vm()
vm_output = vm.run(input.detach().numpy())
torch.testing.assert_close(expected_output, torch.from_numpy(vm_output))
```

Note that currently `vollo-torch` does not accept PyTorch models with multiple
input arguments, so if you need this feature consider whether you can
concatenate your inputs into a single `Tensor` and unweave the model as in this
example.
