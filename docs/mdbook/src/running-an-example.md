# Running an example

The VOLLO SDK contains a trivial program for each accelerator to check if the accelerator is working.

1. Ensure you have run the setup steps:

   ```sh
   cd <vollo-sdk>
   sudo ./load_kernel_driver.sh
   source setup.sh
   export MYRTLE_LICENSE=<your-license-file>
   ```

2. Compile the C runtime example:

   ```sh
   (cd example; make)
   ```

3. Run the example.

   For a block-size 64 accelerator such as `vollo-ia840f-c3b64.jic`:

   ```sh
   ./example/vollo-example example/identity_b64.vollo
   ```

   For a block-size 32 accelerator such as `vollo-ia420f-c6b32.jic`:

   ```sh
   ./example/vollo-example example/identity_b32.vollo
   ```

   You should see an output similar to the following:

   ```sh
   Using program: "example/identity_b64.vollo"
   Using vollo-rt version: 0.12.0
   Program metadata:
     1 input with shape: [128]
     1 output with shape: [128]
   Starting 10000 inferences
   Done
   Ran 10000 inferences in 0.039822 s with:
     mean latency of 3.961121 us
     99% latency of 4.698000 us
     throughput of 251116.318761 inf/s
   ```
