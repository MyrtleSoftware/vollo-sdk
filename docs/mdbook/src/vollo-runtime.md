# Vollo Runtime

The Vollo runtime provides a low latency asynchronous inference API for timing
critical inference requests on the Vollo accelerator.

A couple of example C programs that use the Vollo runtime API have been included in the
installation in the `example/` directory.

In order to use the Vollo runtime you need to have [an accelerator set up](./accelerator-setup.md):

- [A programmed FPGA](./programming-the-fpga.md)
- [A loaded kernel driver and an installed license](./licensing.md)
- Environment set up with `source setup.sh`

## Python API

The Vollo SDK includes Python bindings for the Vollo runtime. These can be more
convenient than the C API for e.g. testing Vollo against PyTorch models.

<!-- markdown-link-check-disable -->

The API for the Python bindings can be found [here](./api-reference/vollo_rt.html).

<!-- markdown-link-check-enable -->

A small example of using the Python bindings is provided [here](./vollo-rt-python-example.md).
