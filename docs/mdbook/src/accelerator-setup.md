# Setting up the Vollo accelerator

This section describes how to program your accelerator card with the Vollo
Accelerator upon first use and how to reprogram your accelerator card with
updated versions of the Vollo Accelerator.
It also describes how to obtain a Vollo license which you will need to use the
Vollo accelerator.

## Environment Variable Setup

The initial setup instructions should be run in the Vollo SDK directory.

```bash
cd vollo-sdk-<VERSION>
```

When using Vollo, you should also have the `setup.sh` script sourced in `bash`
to set up environment variables used by Vollo:

```bash
source setup.sh
```

## Hosts with more than one accelerator

The `vollo-tool` subcommands that inspect or act on a card take an optional
device argument, and act on every Vollo accelerator in the host when it is
omitted. A device is named either by its PCI address or by its index in the
list `vollo-tool device-ids` prints, counting from 0:

```bash
vollo-tool read-hw-config 01:00.0
vollo-tool license-check 0
```

The runtime selects a card the same way: `vollo_rt_add_device` takes the same
form of specifier, where `vollo_rt_add_accelerator` takes an index only. The
`vollo-example` application accepts it as `--device`.
