# Example design

The Vollo IP Core comes with a simple example design.

The example design is for the Bittware IA-420F device - but can be converted to a different device by adapting the relevant files.

The example design contains:

- `rtl`:
  a simple instantiation of `vollo_ip_core` with the input and output AXI4-Stream interfaces are
  hooked up to directly MMIO.
  An efficient design should instead control input and output directly from the FPGA, or if
  control from a CPU host is needed then use a proper DMA instead.
- `bitstream`:
  this is a pre-built bitstream of the provided RTL, it is provided to speed up testing of the runtime
- `runtime`:
  the runtime showcases how to configure and activate the Vollo IP Core using the `vollo-cfg` library.
  It also controls IO to do basic inferences
  (this IO mechanism is only an example, it is *not* optimized for performance)

## Building RTL

A script is provided to automate building the RTL: `vollo_ip_core_example/rtl/build.sh`

Alternatively, you can perform the following steps manually:

- Run QSYS:
  This example project is provided as a TCL generated QSYS project.
  So we need to generate the top level QSYS:

  ```bash
  qsys-script --script=qsys/mcdma_sys.tcl --quartus-project=agilex
  ```

  And generate RTL for that QSYS project:

  ```bash
  qsys-generate mcdma_sys.qsys --synthesis=VERILOG --parallel --quartus-project=agilex.qpf
  ```

- Run Quartus:
  This step can be done manually in the GUI or with the provided `flow.tcl`
  (which runs `syn`, `fit`, `sta`, and `asm`):

  ```bash
  quartus_sh -t flow.tcl
  ```

  In order to synthesise the Vollo IP Core which is encrypted, make sure to add the `vollo-ip-quartus.lic`
  file to your license server, or simply add it to the `LM_LICENSE_FILE` environment variable
  (see [Quartus integration](3-quartus-integration.md)).
