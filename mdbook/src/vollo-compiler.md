# VOLLO Compiler

The VOLLO compiler is made up of 2 Python libraries:

* The `vollo-torch` PyTorch frontend to the compiler.
* The `vollo-python` Python bindings to the backend of the compiler that can
  transform and compile a model to a VOLLO program (`.vollo` file).

The [VOLLO Runtime](vollo-runtime.md) section describes how to run a VOLLO
program on a VOLLO accelerator.
The VOLLO compiler API also includes functionality to simulate and estimate
performance of VOLLO programs.

## Installation

Set up VOLLO environment variables by [sourcing
`setup.sh`](initial-setup.md#environment-variable-setup).

Install the wheel files for the VOLLO compiler libraries. It's recommended that
you install these into a [virtual
environment](https://docs.python.org/3/library/venv.html).

Note: the packaged wheels only support python 3.7 or greater

```bash
> python3 -m venv vollo-venv
> source vollo-venv/bin/activate
> pip install --upgrade pip
> pip install "$VOLLO_SDK"/python/*.whl
```
