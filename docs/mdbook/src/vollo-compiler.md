# Vollo Compiler

The Vollo compiler is made up of 2 Python libraries:

- The `vollo-torch` PyTorch frontend to the compiler.
- The `vollo-compiler` backend that can transform and compile a model to a
  Vollo program (`.vollo` file).

The [Vollo Runtime](vollo-runtime.md) section describes how to run a Vollo
program on a Vollo accelerator.
The Vollo compiler API also includes functionality to simulate and estimate
performance of Vollo programs.

## API Reference

This chapter walks through examples of how to use the Vollo compiler that
should cover the most commonly used parts of the API.

<!-- markdown-link-check-disable -->

A more complete API reference can be found [here](./api-reference).

<!-- markdown-link-check-enable -->

## Installation

Set up Vollo environment variables by [sourcing
`setup.sh`](accelerator-setup.md#environment-variable-setup) in `bash`.

Install the wheel files for the Vollo compiler libraries. It's recommended that
you install these into a [virtual
environment](https://docs.python.org/3/library/venv.html).

Note: the packaged wheels only support python 3.7 or greater

```sh
python3 -m venv vollo-venv
source vollo-venv/bin/activate
pip install --upgrade pip
pip install "$VOLLO_SDK"/python/*.whl
```
