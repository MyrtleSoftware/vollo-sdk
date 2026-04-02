# System Requirements

## CPU Requirements

An x86-64 CPU is required to run the Vollo runtime.

## Accelerator Card Requirements

The SDK runs on a server CPU with PCIe FPGA accelerator cards.
It currently supports the following accelerator cards:

| Accelerator Card     | FPGA                  | Max parameter count |
| -------------------- | -------------------   | ------------------- |
| BittWare IA-420f     | Intel Agilex AGF014   | 3.1 Million         |
| BittWare IA-840f     | Intel Agilex AGF027   | 8.4 Million         |
| Napatech NT400D11    | Intel Agilex AGF014   | 3.1 Million         |
| AMD Alveo V80        | AMD Versal XCV80      | 25.2 Million[^note] |

## Operating System Requirements

Vollo is compatible with Ubuntu 20.04 and later.

[^note]: 50.3 Million if using FP8 weights.
