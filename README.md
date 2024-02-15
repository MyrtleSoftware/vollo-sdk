# VOLLO SDK Readme

##  Programming the FPGA

Please follow the fpga programming instructions in `docs/markdown/programming-the-fpga.md` to setup the Vollo accelerator.

## Load FPGA Kernel Driver


Load the FPGA driver kernel driver:
```bash
sudo ./load-kernel-driver.sh
```

To check if kernel driver is loaded:
```bash
sudo lsmod | grep ifc_uio
```


## Source SDK

To source the Vollo SDK run:
```bash
source setup.sh
```

## Run benchmark script

Install python3 virtual environment:
```bash
sudo apt install python3-venv
```

To run the benchmark script:
```bash
export MYRTLE_LICENSE=<license>
./example/benchmark.sh
```

## User guide

The user guide can be found in markdown at `docs/markdown` or as html:
```bash
open docs/html/index.html
```

## File structure

- *bin* prebuilt applications (vollo-tool)
- *bitstream* FPGA programming files
- *docs* documentation
- *example* example application and benchmark script
- *include* header files
- *lib* library files
- *models* library for generating reference models
- *python* python libraries for the vollo-compiler
