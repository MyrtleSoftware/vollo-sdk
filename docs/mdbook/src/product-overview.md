# Product Overview

The product architecture is shown in the diagram below:

![System Architecture](assets/system-architecture.svg)

Vollo provides a C API to the user, running on the system host CPU.
Vollo targets FPGAs to provide low latency inference.
The FPGA images are prebuilt and included in the product.

Vollo consists of the following elements:

- Vollo Accelerator Bitstream. Programming file for the FPGA on the PCIe
  accelerator card. See the [Accelerator Setup](accelerator-setup.md) section for
  instructions on programming your FPGA with the bitstream.

- [Vollo Compiler](vollo-compiler.md). Compiles ML models defined in PyTorch
  to Vollo programs.

- [Vollo Runtime](vollo-runtime.md). The runtime library for Vollo. It provides
  an asynchronous inference interface for handling input and output for the
  accelerated model.
