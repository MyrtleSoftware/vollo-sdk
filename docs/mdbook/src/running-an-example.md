# Running an example

The Vollo SDK contains a trivial program for each accelerator to check if the accelerator is working.

1. Ensure you have run the setup steps:

   ```sh
   cd <vollo-sdk>
   sudo ./load-kernel-driver.sh
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
   Using vollo-rt version: 18.0.0
   Using Vollo accelerator with 3 core(s) and block_size 64
   Program metadata for model 0:
     1 input with shape: [128]
     1 output with shape: [128]
   Starting 10000 inferences
   Done
   Ran 10000 inferences in 0.020185 s with:
     mean latency of 2.004259 us
     99% latency of 2.176000 us
     throughput of 495411.228723 inf/s
   ```
