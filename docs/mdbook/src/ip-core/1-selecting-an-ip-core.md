# Selecting an IP core configuration

The Vollo IP core is scalable by the number of cores and the size of each core. To identify the best configuration for
your application, we advise the following steps:

1. Test your ML models by running them through the Vollo compiler with different configurations. The virtual machine
   will give you performance metrics for each configuration.

2. Reference the resource usage table below to understand the resource requirements for each configuration.

Note, for initial integration and testing, it is recommended to start with a single core and a small block size. This will
allow for quicker iteration and debugging of the system.

## Compiler support

To test an IP Core configuration with the Vollo compiler, you can create the configuration in python as follows:

```python
import vollo_compiler
ip_core_config = vollo_compiler.Config.ip_core(num_cores = 1, block_size = 32)
```

Once you have received your Vollo IP core, it comes bundled with a Vollo accelerator configuration file in JSON format,
in `vollo_ip_core/vollo-ip-core-config.json`.
This can be loaded into an accelerator configuration in python as follows:

```python
!import os
!os.mkdir("vollo_ip_core")
!vollo_compiler.Config.save(vollo_compiler.Config.ip_core(num_cores = 1, block_size = 32), "vollo_ip_core/vollo-ip-core-config.json")
ip_core_config = vollo_compiler.Config.load("vollo_ip_core/vollo-ip-core-config.json")
!os.remove("vollo_ip_core/vollo-ip-core-config.json")
!os.rmdir("vollo_ip_core")
```

From here, Vollo programs can be generated using the same workflow as described in [example 1](../example-1-mlp.md)
using the config imported above in the call to `to_program`:

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
# Make a pytorch model
input_size = 784
output_size = 10
hidden_size = 128
model = MLP(input_size, output_size, hidden_size)

# Annotate the model with activation shapes
input = torch.randn(input_size)
(model, expected_output) = vollo_torch.fx.prepare_shape(model, input)

# Compile to a Vollo program using the loaded ip_core_config
nnir = vollo_torch.fx.nnir.to_nnir(model)
program = nnir.to_program(ip_core_config)

```

Run the VM to get the cycle count:

```python
vm = program.to_vm()
vm_output = vm.run(input.detach().numpy())
torch.testing.assert_close(expected_output, torch.from_numpy(vm_output), atol = 1e-2, rtol=1e-2)
print("cycle count:", program.cycle_count_per_inference())
```

## Resource usage

The following table shows the resource usage of the Vollo IP Core for different configurations. Note,
these resources may vary depending on the Vollo SDK version.

The block size determines the side of the matrix block. The core scales with the square of this parameter
e.g. a block size 64 core is around 4 times larger than a block size 32 core.

| Cores | Block size | ALMs | M20Ks | DSPs |
| ----- | ---------- | ---- | ----- | ---- |
| 1     | 32         | 43K  | 1084  | 624  |
| 2     | 32         | 78K  | 2000  | 1248 |
| 3     | 32         | 115K | 2932  | 1872 |
| 4     | 32         | 152K | 3880  | 2496 |
| 5     | 32         | 194K | 4844  | 3120 |
| 6     | 32         | 231K | 5824  | 3744 |
| 1     | 64         | 106K | 3065  | 2400 |
| 2     | 64         | 207K | 5840  | 4800 |
| 3     | 64         | 308K | 8631  | 7200 |
