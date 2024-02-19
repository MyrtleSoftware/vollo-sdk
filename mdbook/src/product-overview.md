# Product Overview

The product architecture is shown in the diagram below:

![System Architecture](images/system-architecture.svg)

VOLLO provides a C API to the user, running on the system host CPU.
VOLLO targets FPGAs to provide low latency inference.
The FPGA images are prebuilt and included in the product.

VOLLO consists of the following elements:

* VOLLO Accelerator Bitstream. Programming file for the FPGA on the PCIe
  accelerator card. See the [Initial Setup](initial-setup.md) section for
  instructions on programming your FPGA with the bitstream.

* [VOLLO Compiler](vollo-compiler.md). Compiles ML models defined in PyTorch
  to VOLLO programs.

* [VOLLO Runtime](vollo-runtime.md). The runtime library for VOLLO. It provides
  an asynchronous inference interface for handling input and output for the
  accelerated model.
