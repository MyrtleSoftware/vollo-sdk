# Example 3: Multiple Models in a Vollo Program

Vollo supports putting multiple models on a single accelerator.

Multiple NNIRs can be compiled into a single program:

```python
!import torch
!import torch.nn as nn
!import torch.nn.functional as F
!import vollo_torch
!
!class MLP(nn.Module):
!    def __init__(self, input_size, output_size, hidden_size):
!        super().__init__()
!        self.fc1 = nn.Linear(input_size, hidden_size)
!        self.fc2 = nn.Linear(hidden_size, hidden_size)
!        self.out = nn.Linear(hidden_size, output_size)
!
!    def forward(self, x):
!        x = F.relu(self.fc1(x))
!        residual = x
!        x = F.relu(self.fc2(x)) + residual
!        return self.out(x)
!
# Instantiate an MLP
input_size = 784
output_size = 10
hidden_size = 128
mlp_model = MLP(input_size, output_size, hidden_size)
mlp_input = torch.randn(input_size)
(mlp_model, mlp_expected_output) = vollo_torch.fx.prepare_shape(mlp_model, mlp_input)
mlp_nnir = vollo_torch.fx.nnir.to_nnir(mlp_model)

!class CNN(nn.Module):
!    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size=3):
!        super().__init__()
!        self.conv1 = vollo_torch.nn.PaddedConv1d(in_channels, hidden_channels, kernel_size)
!        self.conv2 = vollo_torch.nn.PaddedConv1d(hidden_channels, out_channels, kernel_size)
!
!    def forward(self, x):
!        x = F.relu(self.conv1(x))
!        x = F.relu(self.conv2(x))
!        return x
!
# Instantiate a CNN
in_channels = 32
out_channels = 1
hidden_channels = 128
cnn_model = CNN(in_channels, out_channels, hidden_channels)

batch_size = 1
sequence_length = 5
cnn_input = torch.randn(batch_size, in_channels, sequence_length)
(cnn_model, cnn_expected_output) = vollo_torch.fx.prepare_shape(cnn_model, cnn_input)
cnn_nnir = vollo_torch.fx.nnir.to_nnir(cnn_model)
(cnn_nnir, output_axis) = cnn_nnir.streaming_transform(2)

# Compile the multi-model program
import vollo_compiler
# Replace the Config in the line below with the Config for the accelerator you
# are using
program_builder = vollo_compiler.ProgramBuilder(vollo_compiler.Config.ia_420f_c6b32())
program_builder.add_nnir(mlp_nnir)
program_builder.add_nnir(cnn_nnir)
multi_model_program = program_builder.to_program()
```

The `vollo_compiler.ProgramBuilder` allows you to create a multi-model program. Building a multi-model program may give an allocation error if
the models can't fit on the given `Config`. Generally each individual model will only have a small latency overhead compared to running it as an individual program. This overhead comes from selecting which model to run.

A `model_index` can be provided when running inferences on the accelerator or on the VM. The models appear in the order in which they were added to the `ProgramBuilder`. For example on the VM:

```python
vm = multi_model_program.to_vm()

mlp_vm_output = vm.run(mlp_input.detach().numpy(), model_index = 0)
torch.testing.assert_close(mlp_expected_output, torch.from_numpy(mlp_vm_output), atol = 1e-2, rtol = 1e-2)

cnn_vm_outputs = []
for i in range(5):
    cnn_vm_outputs.append(vm.run(cnn_input[:, :, i].detach().numpy(), model_index = 1))

torch.testing.assert_close(
    cnn_expected_output,
    torch.stack(
        [torch.from_numpy(output) for output in cnn_vm_outputs],
        axis=output_axis,
    ),
    atol = 1e-2,
    rtol = 1e-2
)
```
