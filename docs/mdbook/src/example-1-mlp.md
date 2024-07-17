# Example 1: MLP

Basic models like multilayer perceptrons (MLP) can be defined without any
changes from a standard PyTorch definition.

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
```

The first stage of compiling a model is to lower it to NNIR.
NNIR is the Vollo compiler's intermediate representation for representing neural
network graphs.
NNIR sits at a similar level of abstraction to ONNX, with most NNIR operators
having direct ONNX or PyTorch analogues.

```python
import vollo_torch

# An input to the model needs to be provided so that its execution can be
# traced
input = torch.randn(input_size)
# Trace the model's execution to annotate it with activation shapes
(model, expected_output) = vollo_torch.fx.prepare_shape(model, input)
nnir = vollo_torch.fx.nnir.to_nnir(model)
```

NNIR can be compiled to a Vollo program given a Vollo accelerator configuration.

```python
import vollo_compiler

config = vollo_compiler.Config.ia_420f_c6b32()
program = nnir.to_program(config)
```

Vollo programs have all their memory allocated statically.
You can print the static resource usage of a program like this:

```python
print(program.metrics())
```

Save the program to a file so that it can be used for inference by the [Vollo
runtime](vollo-runtime.md).

```python
program.save('mlp.vollo')
```

## Simulation

The Vollo compiler can be used to simulate programs in the Vollo virtual machine
(VM).
This is an instruction level simulation of the Vollo accelerator which can be
used to:

- Estimate performance of a model.
  The VM is not cycle accurate but provides an indicative cycle count of a
  model.
- Verify the correctness of the compilation stages, including the effect of
  quantisation.
  Note the output of the VM is not bit accurate to the Vollo accelerator.

Construct a VM instance with your program loaded.
Run the VM by passing it a numpy array of the input.
It should produce the same result as the source PyTorch model, within some
range of floating point error.

```python
vm = program.to_vm()
vm_output = vm.run(input.detach().numpy())
torch.testing.assert_close(expected_output, torch.from_numpy(vm_output))
print("cycle count:", vm.cycle_count())
# Translate the estimated cycle count to a duration for the compute (not
# including IO) in microseconds, using the bitstream clock speed (320 MHz)
print(f"latency (compute): {vm.compute_duration_us():.1f}us")
```

The VM records the number of cycles the program took to execute.
Note there will be some discrepancy between the VM's cycle count and the true
cycle count, so the VM's cycle count should be treated as an estimate.
Also note that the VM does not model the latency of the communication between
the host and the Vollo accelerator.
