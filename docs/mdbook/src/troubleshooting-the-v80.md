# Troubleshooting the V80 / V80LL card

This section provides suggested solutions to common problems that are faced when working with the V80 or V80LL.
Though only the V80 is referenced in the solutions below, they all apply to both the V80 and V80LL cards.

## General troubleshooting

Before trying any other solutions, it's always good practice to power down the host machine and then power it back up again.
This can solve configuration issues related to PCIe enumeration and can solve many other problems.

## Problems programming the V80 over PCIe

The processes below assume that the V80 has already been successfully programmed with a `Vollo` image at least once,
either via JTAG or PCIe. If you have not yet programmed the card with a `Vollo` image, please follow the instructions
in [Programming the V80 via JTAG](./programming-the-v80.md#programming-the-fpga-via-jtag).

If you are having trouble while programming the V80 card over PCIe, or if you cannot access sensor data
from the card, then the first step is to ensure that the `ami` driver is built and loaded correctly.

```sh
cd $VOLLO_SDK/ami_kernel_driver
make
sudo insmod ami.ko
```

If the driver has already been loaded, try unloading and reloading the driver:

```sh
cd $VOLLO_SDK/ami_kernel_driver
sudo rmmod ami
sudo insmod ami.ko
```

Now that the `ami` driver is correctly loaded, the driver association with the v80 should be visible with an `lspci` command.

```sh
lspci -vvvnn -s $BDF
```

If you don't know what your card's BDF is, it may be found by running the command below.

```sh
lspci | grep "Myrtle"
```

Running the full `lspci` command should display the V80 card loaded with a `Vollo` image as below. The card's BDF here is `01:00`.

```sh
01:00.0 Processing accelerators [1200]: Myrtle.ai Device [1ed9:000a]
      Subsystem: Xilinx Corporation Device [10ee:50b4]
      Control: I/O- Mem+ BusMaster- SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx-
      Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
      IOMMU group: 12
      Region 0: Memory at c0000000 (32-bit, non-prefetchable) [size=256M]
      Region 2: Memory at d0000000 (32-bit, non-prefetchable) [size=4M]
      Region 4: Memory at fce0000000 (64-bit, prefetchable) [size=4M]
      Capabilities: <access denied>

01:00.1 Processing accelerators [1200]: Myrtle.ai Device [1ed9:100a]
        Subsystem: Xilinx Corporation Device [10ee:50b4]
        Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx-
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        IOMMU group: 13
        Region 0: Memory at fcd0000000 (64-bit, prefetchable) [size=256M]
        Capabilities: <access denied>
        Kernel driver in use: ami
```

The last line shows the `ami` driver correctly associated with the `1st` physical function of the board, `01:00.1`.

This provides access to the OSPI programming functions available on the V80 via PCIe and the normal PCIe programming
detailed in [Programming the V80 over PCIe](./programming-the-v80.md#programming-the-fpga-over-pcie) should be possible. If not, please continue with
this guide.

## Recovering the V80 from a "bad state"

The V80 card can sometimes enter into a "bad state". This can result in the card not functioning correctly despite following
other troubleshooting steps, including powering down and powering up the host machine.

Such a failure might manifest as follows:

If, while running the PCIe programming commands found in step 2 of [Programming the FPGA over PCIe](./programming-the-v80.md#programming-the-fpga-over-pcie),
you see a failure that displays as below, then it's possible that the V80 card is in a bad state.

```sh
----------------------------------------------
Device | 01:00.1
----------------------------------------------
Current Configuration
----------------------------------------------
UUID   | 611196598b26d6e6896b650e6357e1ba
----------------------------------------------
Incoming Configuration
----------------------------------------------
UUID      | 611196598b26d6e6896b650e6357e1ba
Path      | /home/stuart/Projects/defaultBuilds/v80ll-sensors-amc-update/pdi/amd_v80_gen5x8_nofpt.pdi
Partition | 0
----------------------------------------------
Are you sure you wish to proceed? [Y/n]: Y
[....................................................................................................] 0% | Error: could not program image
EIO: File could not be read or written [do_image_download:154 - errno 19 (No such device)].
```

The suggested solution for stateful failures like this is to:

1. Power the host machine down
2. Completely remove power from the host machine by either unplugging or removing the card from the host machine completely.
3. Leave the host machine and/or V80 card unpowered for around 30 minutes.
4. Reinsert the V80 into the host machine (if removed), reconnect power to the host machine.
5. Power up the host machine again.
6. Re-attempt PCIe programming following the steps provided in [Programming the FPGA over PCIe](./programming-the-v80.md#programming-the-fpga-over-pcie).

This unorthodox solution has been found to work in many cases and is indeed the recommended solution to these problems as provided by AMD [here](https://adaptivesupport.amd.com/s/question/0D5KZ00000aZz790AC/unable-to-recover-the-v80-fpga?language=en_US).
