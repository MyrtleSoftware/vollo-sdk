# VOLLO SDK

The VOLLO SDK is designed for low latency streaming inference of machine
learning (ML) models on FPGA platforms.
The latest SDK is available for download from <https://github.com/MyrtleSoftware/vollo-sdk/releases>.

## User guide

The user guide is available online at <https://vollo.myrtle.ai/>.

It is also included locally as markdown at [docs/mdbook/src](docs/mdbook/src) or in the release as html:

```bash
open docs/html/index.html
```

## Overview

The product architecture is shown in the diagram below:

![System Architecture](docs/mdbook/src/assets/system-architecture.svg)

VOLLO provides a C API to the user, running on the system host CPU.
VOLLO targets FPGAs to provide low latency inference.
The FPGA images (bitstreams) are prebuilt and included in the product.

VOLLO consists of the following elements:

- VOLLO Accelerator Bitstream. Programming file for the FPGA on the PCIe
  accelerator card. See the [Accelerator Setup](https://vollo.myrtle.ai/accelerator-setup.html) section for
  instructions on programming your FPGA with the bitstream.

- [VOLLO Compiler](https://vollo.myrtle.ai/vollo-compiler.html). Compiles ML models defined in PyTorch
  or ONNX to VOLLO programs.

- [VOLLO Runtime](https://vollo.myrtle.ai/vollo-runtime.html). The runtime library for VOLLO. It
  sets up the the VOLLO accelerator with a program and provides an asynchronous inference interface
  for handling input and output for the accelerated model.

## VOLLO Compiler

The VOLLO compiler is available to use without an accelerator card or license.
Along with compiling machine learning models for the VOLLO accelerator,
it also provides a simulation of the accelerator which can be used to provide a performance estimate.

Refer to the [VOLLO Compiler](https://vollo.myrtle.ai/vollo-compiler.html)
section in the user guide and its example walkthroughs to get started with the
compiler.

## Release file structure

| Directory        | Contents                                           |
| ---------------- | -------------------------------------------------- |
| `bin/`           | Prebuilt applications (`vollo-tool`, `vollo-onnx`) |
| `bitstream/`     | FPGA programming files                             |
| `docs/`          | Documentation                                      |
| `example/`       | Example application and benchmark script           |
| `include/`       | Vollo runtime C/C++ header files                   |
| `kernel_driver/` | Kernel driver for Vollo accelerator card           |
| `lib/`           | Vollo runtime shared/static library files          |
| `python/`        | Vollo compiler Python libraries                    |

## Contact

For support and feature requests, please contact <vollo@myrtle.ai>.
