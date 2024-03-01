# Vollo Runtime

The Vollo runtime provides a low latency asynchronous inference API for timing
critical inference requests on the Vollo accelerator.

A couple of example C programs that use the Vollo runtime API have been included in the
installation in the `example/` directory.

In order to use the Vollo runtime you need to have [an accelerator set up](./accelerator-setup.md):
- [A programmed FPGA](./programming-the-fpga.md)
- [A loaded kernel driver and an installed license](./installing-a-license.md)
