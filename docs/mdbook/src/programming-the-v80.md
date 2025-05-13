# Programming the V80 FPGA

## Download the bitstream for your FPGA

The bitstream is available on the [Github Release page] alongside the Vollo SDK. For example to
download the bitstream for the AMD `v80` board with the `c6b32` configuration of Vollo:

[Github Release page]: https://github.com/MyrtleSoftware/vollo-sdk/releases/

```sh
curl -LO https://github.com/MyrtleSoftware/vollo-sdk/releases/download/v23.0.0/vollo-amd-v80-c6b32-23.0.tar.gz
mkdir -p $VOLLO_SDK/bitstream
tar -xzf vollo-amd-v80-c6b32-23.0.tar.gz -C $VOLLO_SDK/bitstream
```

## Programming the FPGA over PCIe

The AMD V80 boards should come pre-programmed with an image that lets you program the flash over
PCIe. First check that the board enumerates correctly:

```sh
$ lspci -d 10ee:50b4
01:00.0 Processing accelerators: Xilinx Corporation Device 50b4
```

If the board has not enumerated then you will need to program the board over JTAG. See [Programming
the FPGA via JTAG](#programming-the-fpga-via-jtag) below.

If the board enumerates correctly, you can program the flash over PCIe. This is the preferred method
of programming the board as it is faster than programming over JTAG, and does not require a USB
programming cable or for Vivado to be installed. To do this we will use the `ami_tool` from the
AMD AVED repository.

1. Clone and build the AVED repository. Depending on your system, you may need to install the
   dependencies first. Also, you may need to patch the kernel driver for it to build.

   ```sh
   git clone https://github.com/Xilinx/AVED
   cd AVED/sw/AMI

   # Rocky Linux 8
   sudo dnf install make
   sudo dnf install kernel-devel-$(uname -r)
   sed -i 's/^static char \*devnode(struct device \*dev,/static char *devnode(const struct device *dev,/' driver/ami_cdev.c

   scripts/build.sh
   ```

   If there is an issue with your system, please contact us.

2. Load the kernel driver and check the current bitstream information:

   ```sh
   sudo insmod driver/ami.ko
   app/build/ami_tool overview
   ```

   You should see something like this:

   ```sh
   AMI
   -------------------------------------------------------------
   Version          | 2.3.0  (0)
   Branch
   Hash             | 0bab29e568f64a25f17425c0ffd1c0e89609b6d1
   Hash Date        | 20240307
   Driver Version   | 2.3.0  (0)


   BDF       | Device          | UUID                               | AMC          | State
   -----------------------------------------------------------------------------------------
   01:00.0   | Alveo V80 ES3   | e8134e38dcac1a63e93bfb1b320dd588   | 2.3.0  (0)   | READY
   ```

   The UUID here is a hash of the source used to build the bitstream. Your UUID might be different
   but as long as version of the AMC and AMI versions match it should be fine. If the board is not
   in the `READY` state, power cycle the host and try again.

   The AMI tool can also be used to show the sensor information, such as temperature and power.

   ```sh
   app/build/ami_tool sensors
   ```

   The `ami` kernel module is needed for the `ami_tool` to work but it is not needed for the Vollo
   runtime.

3. Program the flash with the `ami_tool`. There will be a progress bar and it should take around 5
   minutes to program the flash. **Replace `01:00.0` with the BDF of your device:**

   ```sh
   sudo app/build/ami_tool cfgmem_program -d 01:00.0 -t primary -i $VOLLO_SDK/bitstream/vollo-amd-v80-c6b32.pdi -p 0 -y
   ```

   There will be a progress bar and it should take around 5 minutes to program the flash. You will
   need to power cycle the host for the new bitstream to be loaded.

   <div class="warning">
   Sometimes a V80 host machine will hang on boot. You may need to force another power cycle of the
   host to bring it back. Occasionally a power cycle isn't enough and you may need to turn the power
   off for several minutes before turning it back on.
   </div>

4. You can check a Vollo bitstream is loaded with lspci (note the `b5d4` in the subsystem device id on PF1):

   ```sh
   $ lspci -d 10ee: -knn
   01:00.0 Processing accelerators [1200]: Xilinx Corporation Device [10ee:50b4]
           Subsystem: Xilinx Corporation Device [10ee:000e]
   01:00.1 Processing accelerators [1200]: Xilinx Corporation Device [10ee:50b5]
           Subsystem: Xilinx Corporation Device [10ee:b5d4]
   ```

   You can also run `ami_tool overview` and check that the V80 is in the `READY` state and that
   the UUID matches the `hash` in `vollo-amd-v80-c6b32.json`.

   ```sh
   sudo insmod driver/ami.ko
   app/build/ami_tool overview
   cat "$VOLLO_SDK"/bitstream/vollo-amd-v80-c6b32.json
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

3. Check a Vollo bitstream is loaded (note the `b5d4` in the subsystem device id on PF1):

   ```sh
   $ lspci -d 10ee: -knn
   01:00.0 Processing accelerators [1200]: Xilinx Corporation Device [10ee:50b4]
           Subsystem: Xilinx Corporation Device [10ee:000e]
   01:00.1 Processing accelerators [1200]: Xilinx Corporation Device [10ee:50b5]
           Subsystem: Xilinx Corporation Device [10ee:b5d4]
   ```
