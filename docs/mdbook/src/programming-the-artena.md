# Programming the Artena FPGA

This section assumes that the Vollo SDK is already installed and setup on the machine that you are using. If you
haven't done so already, instructions for how to do that may be found at: [Vollo SDK Installation](https://vollo.myrtle.ai/latest/installation.html)

Make sure the `VOLLO_SDK` environment variable is set by sourcing setup.sh from the Vollo SDK.

```bash
source <path-to-VOLLO_SDK>/setup.sh
```

## Download the bitstream for your FPGA

The bitstream is available on the [Github Release page] alongside the Vollo SDK. For example to
download the bitstream for the AMD `Artena` board with the `c8b32` configuration of Vollo:

[Github Release page]: https://github.com/MyrtleSoftware/vollo-sdk/releases/

```sh
curl -LO https://github.com/MyrtleSoftware/vollo-sdk/releases/download/v27.0.1/vollo-amd-artena-c8b32-27.0.tar.gz
mkdir -p $VOLLO_SDK/bitstream
tar -xzf vollo-amd-artena-c8b32-27.0.tar.gz -C $VOLLO_SDK/bitstream
```

An LSTM-only image is available for the Artena which runs at a slightly higher clock frequency than the full-featured-Vollo
c8b32 version. This may be downloaded as follows:

```sh
curl -LO https://github.com/MyrtleSoftware/vollo-sdk/releases/download/v27.0.1/vollo-amd-artena-c8b32lstm-27.0.tar.gz
mkdir -p $VOLLO_SDK/bitstream
tar -xzf vollo-amd-artena-c8b32lstm-27.0.tar.gz -C $VOLLO_SDK/bitstream
```

## Programming the FPGA via JTAG

Programming an Artena board over JTAG is necessary if the board does not yet have a Vollo image loaded on it
or if the device does not enumerate correctly. [Programming over PCIe](#programming-the-fpga-over-pcie) is
preferred. If the board does not enumerate or there is some other issue with PCIe programming then JTAG
programming is the only option.

This requires a USB cable to be connected to the accelerator card and Vivado to be installed on the
system so that the device can be programmed over JTAG.

[download page]: https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools.html

1. Download and install Vivado Lab Edition:

    - Navigate to the Vivado Design Tools [download page].
    - Under "Vivado Lab Solutions" find "Vivado 2025.2: Lab Edition - Linux (TAR/GZIP - 1.99 GB)" (later versions may be available).
    - Download the file and extract it to a directory of your choice. You will need an AMD account
      to download the file. You can create an account for free.
    - Pick a location to install `Vivado_Lab`, e.g. `/opt/Xilinx`, a user directory like `~/Xilinx` is
      also fine:

      ```sh
      VIVADO_DIR=~/Xilinx
      mkdir -p $VIVADO_DIR
      ```

    - Extract the tarball:

      ```sh
      tar xf Vivado_Lab_Lin_2025.2_1114_2157.tar
      cd Vivado_Lab_Lin_2025.2_1114_2157
      ```

    - Run the installer:

      ```sh
      ./xsetup --agree 3rdPartyEULA,XilinxEULA --batch Install --edition "Vivado Lab Edition (Standalone)" --location $VIVADO_DIR
      ```

    - Check that installation was successful:

      ```sh
      $ $VIVADO_DIR/2025.2/Vivado_Lab//bin/vivado_lab -version
      Vivado Lab Edition v2025.2 (64-bit)
      SW Build 6299465 on Fri Nov 14 21:19:43 MST 2025
      Tool Version Limit: 2025.11
      Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
      Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
      ```

2. Run the `flash_vollo-amd-artena-c8b32.tcl` script to program the Artena board:

    ```sh
    sudo $VIVADO_DIR/2025.2/Vivado_Lab/bin/vivado_lab -mode batch -source ./flash_vollo-amd-artena-c8b32.tcl
    ```

   This prints out a lot of lines while programming and takes about 10 minutes.

   If you get an error like this:

   ```sh
   ERROR: [Labtoolstcl 44-469] There is no current hw_target.
   ```

   Make sure that you ran `vivado_lab` with `sudo` and that the USB cable is plugged in.

   After programming you must power cycle the host for the new bitstream to be loaded.

3. If successful the device should now enumerate as a Myrtle.ai Vollo device:

   ```sh
   $ lspci -d 1ed9:
   01:00.0 Processing accelerators: Myrtle.ai Device 000a
   01:00.1 Processing accelerators: Myrtle.ai Device 100a
   ```

## Programming the FPGA over PCIe

If your FPGA is already programmed with the Vollo accelerator then you can update the bitstream over
PCIe. You can check if the device is programmed with the Myrtle.ai Vollo bitstream by running:

```sh
$ lspci -d 1ed9:
01:00.0 Processing accelerators: Myrtle.ai Device 000a
01:00.1 Processing accelerators: Myrtle.ai Device 100a
```

If the device has not been programmed with the vollo bitstream then you will need to program the board
over JTAG. See [Programming the FPGA via JTAG](#programming-the-fpga-via-jtag).

The following instructions will program the Vollo bitstream over PCIe:

1. First load the kernel driver.

   ```sh
   sudo ./load-kernel-driver.sh vfio
   ```

2. Once the kernel driver is loaded you can program the flash with `vollo-tool`:

   ```sh
   sudo $VOLLO_SDK/bin/vollo-tool fpga-config overwrite-partition ${device_index:?} $VOLLO_SDK/bitstream/vollo-amd-artena-c8b32.pdi USER_IMAGE
   ```

   The progress will be displayed and it should take a couple of minutes to program the flash. You will
   need to power cycle the host for the new bitstream to be loaded.

3. If successful the device should now enumerate as a Myrtle.ai Vollo device:

   ```sh
   $ lspci -d 1ed9:
   01:00.0 Processing accelerators: Myrtle.ai Device 000a
   01:00.1 Processing accelerators: Myrtle.ai Device 100a
   ```
