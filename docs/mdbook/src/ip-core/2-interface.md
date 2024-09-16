# IP Core Interface

The Vollo IP Core is programmed with a neural network model (Vollo program) via a configuration bus. Once this is done, the IP Core
can run the model by streaming input data to the IP Core and receiving the output data.

![Vollo IP Core](../assets/vollo-ip-core.svg)

The IP Core has the following interfaces:

- Config clock and reset signals. This a clock which is expected to be at a frequency of around 100MHz. It is only used for
  configuration and not for running the model.
- Config bus. This is a 32-bit wide AXI4-Lite bus used to activate the device with a license key and to configure the IP Core. It is synchronous to the config clock.
- Compute clock and reset signals. This is the clock used for running the model. In the example design this clock frequency is
  set to 320MHz.
- Input data bus. This a AXI4-Stream bus used to stream input data to the IP Core. It size varies depending on the size of the
  cores in the IP. For a 32-block size design, this is 512 wide (16 bits per value using brainfloat 16). It is synchronous to the compute clock.
- Model selection bus. This is an AXI4-Stream interface for providing the model index to be run if the IP core has been configured
  with multiple models. If the IP Core has been configured with a single model, providing the index is optional. It is synchronous to the compute clock.
- Output data bus. This a AXI4-Stream bus used to stream output data from the IP Core. It is synchronous to the compute clock.

## Configuration bus

The configuration bus is a 32-bit wide AXI4-Lite bus. The normal rules for AXI4-Lite buses
should be followed with the following exceptions:

- Write strobe: The write strobe should either be fully asserted or fully deasserted. Partially asserted write strobes are not supported.
- The protection signals, `config_awprot` and `config_arprot`, are unused and ignored.

Verilog signals:

```verilog
    // Config interface clock and active-high synchronous reset:
      input  logic          config_clock
    , input  logic          config_reset

    // Config AXI4-Lite interface.
    // The config_awprot and config_arprot inputs are unused
    // and ignored.
    , input  logic          config_awvalid
    , output logic          config_awready
    , input  logic [20:0]   config_awaddr
    , input  logic [2:0]    config_awprot

    , input  logic          config_wvalid
    , output logic          config_wready
    , input  logic [31:0]   config_wdata
    , input  logic [3:0]    config_wstrb

    , input  logic          config_arvalid
    , output logic          config_arready
    , input  logic [20:0]   config_araddr
    , input  logic [2:0]    config_arprot

    , output logic          config_rvalid
    , input  logic          config_rready
    , output logic [31:0]   config_rdata
    , output logic [1:0]    config_rresp

    , output logic          config_bvalid
    , input  logic          config_bready
    , output logic [1:0]    config_bresp

```

The AXI4-Lite interface is used to configure the Vollo IP Core and to do so must be accessible from
the host system. The configuration can then be done by providing functions to communicate with the
bus to [Vollo configuration API](4-config.md).

## Input and Output Streams

The input and output streams are AXI4-Stream interfaces. The input and output are packed as flattened tensor and
padded to the next multiple of block-size. The data should be packed in *little-endian* format. The output stream
includes `tkeep` and `tlast` signals to indicate when the end of the packet and which bytes are valid (i.e. not padding).

For example, an input of tensor dimension `[62]` to an ip-core with block size 32 should be provided as two
words, the first with a full 32 brainfloat values, and the second with the remaining 30 brainfloat values and 2 padding values.
They should be packed as follows:

| Word | 511:496     | 495:480     | 479:464     | ... | 31:16       | 15:0        |
| ---- | ----------- | ----------- | ----------- | --- | ----------- | ----------- |
| 0    | `input[31]` | `input[30]` | `input[29]` | ... | `input[1]`  | `input[0]`  |
| 1    | `X`         | `X`         | `input[61]` | ... | `input[33]` | `input[32]` |

Verilog signals:

```verilog

    // Core clock and active-high synchronous reset:
    , input  logic          compute_clock
    , input  logic          compute_reset

    // Input AXI4-Stream interface:
    , input  logic          input_tvalid
    , output logic          input_tready
    , input  logic [511:0]  input_tdata

    // Output AXI4-Stream interface:
    , output logic          output_tvalid
    , input  logic          output_tready
    , output logic          output_tlast
    , output logic [63:0]   output_tkeep
    , output logic [511:0]  output_tdat

```

## Model Selection

> :warning: The IP Core does not currently support multiple models. This feature is planned for a future release.
> The model selection bus is ignored and can be driven with any value.

The model selection bus can be used to select between multiple models that have been configured into the IP Core. This selection
can be provided ahead of the data stream. The model selection bus is a 16-bit wide AXI4-Stream interface.

For single model Vollo programs, it is not required to drive the model selection bus, however the IP Core
will accept the value if it is driven.

```verilog

    // Model select AXI4-Stream interface:
    , input  logic         model_select_tvalid
    , output logic         model_select_tready
    , input  logic [15:0]  model_select_tdata

```

## Verilog instantiation

The complete component interface is as follows:

```sv

module vollo_ip_core
  (
    // Config interface clock and active-high synchronous reset:
      input  logic          config_clock
    , input  logic          config_reset

    // Config AXI4-Lite interface.
    // The config_awprot and config_arprot inputs are unused
    // and ignored.
    , input  logic          config_awvalid
    , output logic          config_awready
    , input  logic [20:0]   config_awaddr
    , input  logic [2:0]    config_awprot

    , input  logic          config_wvalid
    , output logic          config_wready
    , input  logic [31:0]   config_wdata
    , input  logic [3:0]    config_wstrb

    , input  logic          config_arvalid
    , output logic          config_arready
    , input  logic [20:0]   config_araddr
    , input  logic [2:0]    config_arprot

    , output logic          config_rvalid
    , input  logic          config_rready
    , output logic [31:0]   config_rdata
    , output logic [1:0]    config_rresp

    , output logic          config_bvalid
    , input  logic          config_bready
    , output logic [1:0]    config_bresp

    // Core clock and active-high synchronous reset:
    , input  logic          compute_clock
    , input  logic          compute_reset

    // Model select AXI4-Stream interface:
    , input  logic          model_select_tvalid
    , output logic          model_select_tready
    , input  logic [15:0]   model_select_tdata

    // Input AXI4-Stream interface:
    , input  logic          input_tvalid
    , output logic          input_tready
    , input  logic [511:0]  input_tdata

    // Output AXI4-Stream interface:
    , output logic          output_tvalid
    , input  logic          output_tready
    , output logic          output_tlast
    , output logic [63:0]   output_tkeep
    , output logic [511:0]  output_tdata
  );

```
