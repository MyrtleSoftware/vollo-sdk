# Vollo SDK

The Vollo SDK is designed for low latency streaming inference of machine
learning (ML) models on FPGA platforms.

## Installation

The latest SDK is available for download from the [GitHub Releases page][GitHub Releases].

Download the `vollo-sdk-<version>.run` self-extractable archive and execute it
to extract the Vollo SDK [contents](#release-file-structure) to the current
directory:

```sh
chmod +x vollo-sdk-<version>.run
./vollo-sdk-<version>.run
```

The FPGA images, e.g. `vollo-ia420f-c6b32-<version>.tar.gz`, are downloadable
as separate files, also from the [GitHub Releases page][GitHub Releases].

[GitHub Releases]: https://github.com/MyrtleSoftware/vollo-sdk/releases

## User guide

The user guide is available online at <https://vollo.myrtle.ai/>.

It is also included locally as markdown at [docs/mdbook/src](docs/mdbook/src) or in the release as html:

```bash
open docs/html/index.html
```

## Overview

The product architecture is shown in the diagram below:

![System Architecture](docs/mdbook/src/assets/system-architecture.svg)

Vollo provides a C API to the user, running on the system host CPU.
Vollo targets FPGAs to provide low latency inference.
The FPGA images (bitstreams) are prebuilt and included in the product.

Vollo consists of the following elements:

- Vollo Accelerator Bitstream. Programming file for the FPGA on the PCIe
  accelerator card. See the [Accelerator
  Setup](https://vollo.myrtle.ai/latest/accelerator-setup.html) section for
  instructions on programming your FPGA with the bitstream.

- [Vollo Compiler](https://vollo.myrtle.ai/latest/vollo-compiler.html).
  Compiles ML models defined in PyTorch or ONNX to Vollo programs.

- [Vollo Runtime](https://vollo.myrtle.ai/latest/vollo-runtime.html). The
  runtime library for Vollo. It sets up the the Vollo accelerator with a
  program and provides an asynchronous inference interface for handling input
  and output for the accelerated model.

## Vollo Compiler

The Vollo compiler is available to use without an accelerator card or license.
Along with compiling machine learning models for the Vollo accelerator,
it also provides a simulation of the accelerator which can be used to provide a performance estimate.

Refer to the [Vollo Compiler](https://vollo.myrtle.ai/latest/vollo-compiler.html)
section in the user guide and its example walkthroughs to get started with the
compiler.

## Release file structure

| Directory        | Contents                                           |
| ---------------- | -------------------------------------------------- |
| `bin/`           | Prebuilt applications (`vollo-tool`, `vollo-onnx`) |
| `docs/`          | Documentation                                      |
| `example/`       | Example application and benchmark script           |
| `include/`       | Vollo runtime C/C++ header files                   |
| `kernel_driver/` | Kernel driver for Vollo accelerator card           |
| `lib/`           | Vollo runtime shared/static library files          |
| `python/`        | Vollo compiler Python libraries                    |

## Contact

For support and feature requests, please contact <vollo@myrtle.ai>.
