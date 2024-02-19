# Programming the FPGA

## Programming the FPGA via JTAG

If your FPGA is not already programmed with the VOLLO accelerator then please
follow these instructions to load the bitstream into the accelerator card's
flash memory.

This requires a USB cable to be connected to the accelerator card and Quartus
programmer to be installed on the system so that the device can be programmed
over JTAG.

If the FPGA card already has a VOLLO Accelerator Bitstream, it can be updated
over PCIe. See the section [Program the FPGA via
PCIe](#program-the-fpga-via-pcie) below.
Programming over PCIe is faster, and does not require a USB programming cable or
for Quartus Programmer to be installed.

1. Download and install the latest Quartus Programmer:
   - Navigate to
     <https://www.intel.com/content/www/us/en/software-kit/782411/intel-quartus-prime-pro-edition-design-software-version-23-2-for-linux.html>.
   - Select `Additional Software` and scroll down to find the Programmer.
   - Follow the instructions for installation.

2. Start the jtag daemon:

   ```bash
   > QUARTUS_DIR=/opt/intelFPGA_pro/23.2.0.94/qprogrammer/quartus
   > sudo killall jtagd
   > sudo $QUARTUS_DIR/bin/jtagd
   ```

3. Run `jtagconfig` from the Quartus install, you should see the device(s):

   ```bash
   > $QUARTUS_DIR/bin/jtagconfig
   1) IA-840F [1-5.2]
     0341B0DD   AGFB027R25A(.|R0)
   ```

4. Navigate to the directory containing the `jic` file:

   ```bash
   > cd <vollo-sdk>/bitstream
   ```

5. Set the JTAG clock frequency of the device you want to program to 16 MHz.
   Specify the device by providing the name returned by `jtagconfig`:

   ```bash
   > $QUARTUS_DIR/bin/jtagconfig --setparam "IA-840F [1-5.2]" JtagClock 16M
   ```

6. Start the programming operation on the chosen device. This takes around 20
   minutes. For the IA840F:

   ```bash
   > $QUARTUS_DIR/bin/quartus_pgm -c "IA-840F [1-5.2]" -m JTAG -o "ipv;vollo-ia840f.jic"
   ```

   Or for IA420F:

   ```bash
   > $QUARTUS_DIR/bin/quartus_pgm -c "IA-420F [1-5.2]" -m JTAG -o "ipv;vollo-ia420f.jic"
   ```

7. Go back to 5 and program any other devices.

8. Power off the system and start it back up. The bitstream will now be loaded
   onto the FPGA.

9. Check the VOLLO bitstream is loaded:

   ```bash
   > lspci -d 1ed9:766f
   51:00.0 Processing accelerators: Myrtle.ai Device 766f (rev 01)
   ```

## Programming the FPGA via PCIe

NOTE: this can only be done with an FPGA that is already programmed with a VOLLO bitstream.

1. Load the kernel driver:

   ```bash
   > sudo ./load-kernel-driver.sh
   ```

2. Check the current bitstream information:

   ```bash
   > bin/vollo-tool bitstream-info
   ```

3. Check that the device is set up for remote system updates by running the
   command below, with `device index` representing the index of the device you
   want to update, in the order shown in the previous command, starting from 0.
   It should print a `json` string to the terminal showing the device status.

   ```bash
   > bin/vollo-tool fpga-config rsu-status <device index>
   ```

4. Update the `USER_IMAGE` partition of the flash with the new bitstream image
   contained in the `rpd` archive. This should take around 5 minutes. Do not
   interrupt this process until it completes.

   ```bash
   > bin/vollo-tool fpga-config overwrite-partition <device index> <.rpd.tar.gz file> USER_IMAGE
   ```

5. Repeat step 4 for any other devices you wish to update.

6. Power off the system and start it back up.

7. Repeat steps 1, 2 and 3. The `bitstream-info` command should show that the
   updated bitstream has been loaded (e.g. a newer release date), and the output
   of the `rsu-status` command should show all zeroes for the `error_code` and
   `failing_image_address` fields.
