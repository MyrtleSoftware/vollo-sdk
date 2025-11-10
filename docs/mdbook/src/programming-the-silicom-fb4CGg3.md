# Programming the Silicom-fb4CGg3 FPGA

## Download the bitstream for your FPGA

The bitstream is available on the [Github Release page] alongside the Vollo SDK. For example to
download the bitstream for the Silicom `fb4CGg3@VU09P-3` board with the `c3b32` configuration of
Vollo:

[Github Release page]: https://github.com/MyrtleSoftware/vollo-sdk/releases/

```sh
curl -LO https://github.com/MyrtleSoftware/vollo-sdk/releases/download/v25.1.1/vollo-silicom-fb4CGg3@VU09P-3-c3b32-25.1.tar.gz
mkdir -p $VOLLO_SDK/bitstream
tar -xzf vollo-silicom-fb4CGg3@VU09P-3-c3b32-25.1.tar.gz -C $VOLLO_SDK/bitstream
```

## Programming the FPGA over PCIe

Unless something has gone wrong, you should always be able to program the FPGA over PCIe.

You can check if the device is programmed with a Myrtle.ai Vollo bitstream by running:

```sh
$ lspci -d 1ed9:
01:00.0 Processing accelerators: Myrtle.ai Device 000a
```

If the device has not been programmed with the Vollo bitstream then it likely has the factory image
which is also suitable for programming over PCIe:

```sh
$ lspci -d 1c2c:
01:00.0 Ethernet controller: Silicom Denmark Device 0001
```

If the device is not enumerating at all please contact us.

The following instructions will program the Vollo bitstream over PCIe:

1. First load the kernel driver.

   ```sh
   sudo ./load-kernel-driver.sh vfio
   ```

   There may be compilation issues with your version of Linux. This has been checked with Rocky Linux
   8.10. If there is an issue with your system, please contact us.

2. Once the kernel driver is loaded you can program the flash with `vollo-tool`:

   ```sh
   sudo $VOLLO_SDK/bin/vollo-tool fpga-config overwrite-partition ${device_index:?} $VOLLO_SDK/bitstream/vollo-silicom-fb4CGg3@VU09P-3-c3b32.bit USER_IMAGE
   ```

   The progress will be displayed and it should take around 5 minutes to program the flash. You will
   need to power cycle the host for the new bitstream to be loaded.

3. If successful the device should now enumerate as a Myrtle.ai Vollo device:

   ```sh
   $ lspci -d 1ed9:
   01:00.0 Processing accelerators: Myrtle.ai Device 000a
   ```
