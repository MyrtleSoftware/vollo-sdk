# Quartus Integration

## Quartus license for Encrypted RTL

Add the quartus license file, `vollo_ip_core/vollo-ip-quartus.lic`, to Quartus to decrypt the Vollo IP core. You can either:

1. Add the license file to your license server.

2. Add the license file locally within Quartus by setting the environment variable `LM_LICENSE_FILE` or adding
   the license file in the GUI `Tools > License Setup`.

## Sourcing design

The following files need sourcing:
- `vollo_ip_core/vollo_ip_core.sv` the top-level interface.
- `vollo_ip_core/vollo_ip_core_internal.vqm` the encrypted RTL design.
- The memory initialization files, `.mif`, in `vollo_ip_core`.

This can be done in the GUI or by using the following `tcl` commands:
```tcl
require file_util

set_global_assignment -name SYSTEM_VERILOG_FILE vollo_ip_core/vollo_ip_core.sv
set_global_assignment -name VQM_FILE vollo_ip_core/vollo_ip_core_internal.vqm
foreach file [fileutil::findByPattern vollo_ip_core *.mif] {
  set_global_assignment -name MIF_FILE $file
}
```
