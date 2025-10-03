# Programming the V80 FPGA

## Download the bitstream for your FPGA

The bitstream is available on the [Github Release page] alongside the Vollo SDK. For example to
download the bitstream for the AMD `v80` board with the `c6b32` configuration of Vollo:

[Github Release page]: https://github.com/MyrtleSoftware/vollo-sdk/releases/

```sh
curl -LO https://github.com/MyrtleSoftware/vollo-sdk/releases/download/v25.1.0/vollo-amd-v80-c6b32-25.1.tar.gz
mkdir -p $VOLLO_SDK/bitstream
tar -xzf vollo-amd-v80-c6b32-25.1.tar.gz -C $VOLLO_SDK/bitstream
```

## Programming the FPGA over PCIe

If your FPGA is already programmed with the Vollo accelerator then you can update the bitstream over
PCIe. You can check if the device is programmed with the Myrtle.ai Vollo bitstream by running:

```sh
$ lspci -d 1ed9:
01:00.0 Processing accelerators: Myrtle.ai Device 100a
01:00.1 Processing accelerators: Myrtle.ai Device 000a
```

If the device has not been programmed with the vollo bitstream then you will need to program the board
over JTAG. See [Programming the FPGA via JTAG](#programming-the-fpga-via-jtag) below.

Programming over PCIe is the preferred method of programming the board as it is faster than
programming over JTAG, and does not require a USB programming cable or for Vivado to be installed.

1. Build and insert the ami driver.

   ```sh
   cd ami_kernel_driver
   make
   sudo insmod ami.ko
   ```

   There may be compilation issues with your version of Linux. This has been checked with Rocky Linux
   8.10. If there is an issue with your system, please contact us.

2. Once the kernel driver is loaded you can program the flash with `vollo-tool` (which uses
   `ami_tool`). If you only have one board, `device_index` is `0`.

   ```sh
   sudo $VOLLO_SDK/bin/vollo-tool fpga-config overwrite-partition ${device_index:?} $VOLLO_SDK/bitstream/vollo-amd-v80-c6b32.pdi USER_IMAGE
   ```

   There will be a progress bar and it should take around 5 minutes to program the flash. You will
   need to power cycle the host for the new bitstream to be loaded.

   <div class="warning">
   Sometimes a V80 host machine will hang on boot. You may need to force another power cycle of the
   host to bring it back. Occasionally a power cycle isn't enough and you may need to turn the power
   off for several minutes before turning it back on.
   </div>

3. If successful the device should now enumerate as a Myrtle.ai Vollo device:

   ```sh
   $ lspci -d 1ed9:
   01:00.0 Processing accelerators: Myrtle.ai Device 100a
   01:00.1 Processing accelerators: Myrtle.ai Device 000a
   ```

## Programming the FPGA via JTAG

Programming a V80 board over JTAG is only needed if the board does not enumerate correctly.
Programming over PCIe is preferred. If the board does not enumerate or there is some other issue
with PCIe programming then JTAG programming is the only option.

This requires a USB cable to be connected to the accelerator card and Vivado to be installed on the
system so that the device can be programmed over JTAG.

[download page]: https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools.html

1. Download and install Vivado Lab Edition:

    - Navigate to the Vivado Design Tools [download page].
    - Under "Vivado Lab Solutions" find "Vivado 2024.2: Lab Edition - Linux (TAR/GZIP - 1.99 GB)" (later versions may be available).
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
      tar xf Vivado_Lab_Lin_2024.2_1113_1001.tar
      cd Vivado_Lab_Lin_2024.2_1113_1001/
      ```

    - Run the installer:

      ```sh
      ./xsetup --agree 3rdPartyEULA,XilinxEULA --batch Install --edition "Vivado Lab Edition (Standalone)" --location $VIVADO_DIR
      ```

    - Check that installation was successful:

      ```sh
      $ $VIVADO_DIR/Vivado_Lab/2024.2/bin/vivado_lab -version
      Vivado Lab Edition v2024.2 (64-bit)
      SW Build 5239620 on Fri Nov 08 16:21:51 MST 2024
      Tool Version Limit: 2024.11
      Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
      Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
      ```

2. Run the `program_v80_fpt.tcl` script to program the V80 board:

    ```sh
    sudo $VIVADO_DIR/Vivado_Lab/2024.2/bin/vivado_lab -mode batch -source ./flash_vollo-amd-v80-c6b32.tcl
    ```

   This prints out a lot of lines while programming and takes about 10 minutes.

   If you get an error like this:

   ```sh
   ERROR: [Labtoolstcl 44-469] There is no current hw_target.
   ```

   Make sure that you ran `vivado_lab` with `sudo` and that the USB cable is plugged in.

   After programming you must power cycle the host for the new bitstream to be loaded.

   <div class="warning">
   Sometimes a V80 host machine will hang on boot. You may need to force another power cycle of the
   host to bring it back. Occasionally a power cycle isn't enough and you may need to turn the power
   off for several minutes before turning it back on.
   </div>

3. If successful the device should now enumerate as a Myrtle.ai Vollo device:

   ```sh
   $ lspci -d 1ed9:
   01:00.0 Processing accelerators: Myrtle.ai Device 100a
   01:00.1 Processing accelerators: Myrtle.ai Device 000a
   ```
