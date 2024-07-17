# Runtime configuration

The Vollo IP Core needs to be programmed with a Vollo program and have its license activated before it can run inferences.
This configuration happens over the AXI4-Lite configuration bus of the Vollo IP Core.

A C library is provided to do this configuration. In order to use the library the user must supply functions for
writing and reading to Vollo IP Core configuration bus. The library then has functions to activate the IP core using
a license and configure the core with a program.

The C API is provided as part of the `VOLLO_SDK`:
- Header is `$VOLLO_SDK/include/vollo-cfg.h`
- Library is `$VOLLO_SDK/lib/libvollo_cfg.so` (or `libvollo_cfg.a` for static linking)

The key concept behind this API is that the user of the Vollo IP Core is in full control of the communication over the configuration bus.
Please refer to the documentation in the header file for details on how to use the API and which considerations should be taken when setting up communication with the configuration bus.

## Licensing a Vollo IP Core

On the first use of an FPGA with the Vollo IP Core, information about the device needs to be gathered in order to acquire a license.
The [section about licensing](../licensing.md) describes how to use `vollo-tool` to get the device ID information.
Unfortunately this method does not work for the Vollo IP Core, as `vollo-tool` does not know how to communicate with the Vollo IP Core.
The C API provides the function `vollo_cfg_print_device_id`, this prints the relevant information to STDOUT using the provided reader and writer to communicate with the IP Core.

Once a license has been acquired, the function `vollo_cfg_activate_license` is used to activate a license (it uses the `MYRTLE_LICENSE` environment variable to find the license file).

## Example design

The Vollo IP Core release contains an example runtime which makes use of this API: `vollo_ip_core_example/runtime/vollo_cfg_example.c`

The example runtime is written to work with the [example](./5-example-design.md) RTL instantiation of the Vollo IP Core which sets up the configuration bus to be accessible over BAR 2 using the ifc_uio kernel driver
(the patched version of the kernel driver provided in `VOLLO_SDK` works just as well as the original Intel provided one for this example)

These are the main steps of the configuration are:
- Acquire and activate a license
- Load a Vollo program onto the Vollo IP Core with `vollo_cfg_load_program`

All of these functions use the same API with a custom reader and writer to communicate with the configuration bus.
The reader and writer are quite generic and take a user controllable context in order to be implementable in most environments (do tell us if this API does not suit your needs).

To build and run the example runtime, please refer to: `vollo_ip_core_example/runtime/README.md`
