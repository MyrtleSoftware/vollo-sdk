# Running the benchmark

The release comes with a benchmark script that can be used to measure the
performance of the accelerator for a variety of models.
The script uses the vollo compiler to compile the models for your accelerator and then runs the
models on the accelerator to measure the performance.

1. Install the script dependencies:

   ```bash
   sudo apt install python3-venv jq
   ```

   Note, the compiler requires python 3.7 or later.

2. Ensure you have run the setup steps:

   ```sh
   cd <vollo-sdk>
   sudo ./load-kernel-driver.sh
   source setup.sh
   export MYRTLE_LICENSE=<your-license-file>
   ```

3. Run the benchmark:

   ```sh
   $VOLLO_SDK/example/benchmark.sh
   ```

   With `RUN_IO_TEST=1` it also measures the [IO round trip](benchmark-io.md) for
   input and output sizes from 64 B to 16 KiB, using the runtime's defaults. It
   writes `work/io-test-metrics.csv`, to compare with the "User buffers" tables, and
   `work/io-test-raw-buffer-metrics.csv`, to compare with the "Raw DMA buffers" tables.

4. You can cross reference your numbers with those in the [benchmarks](benchmark.md) section of the documentation.
