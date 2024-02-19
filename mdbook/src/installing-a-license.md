# Installing a License

1. Load the kernel driver `sudo ./load-kernel-driver.sh`

2. Run `vollo-tool device-ids`, this will enumerate all VOLLO accelerators and
   and output their device IDs.

   ```bash
   > bin/vollo-tool device-ids | tee vollo.devices
   ```

3. Send the `vollo.devices` to `vollo-license@myrtle.ai`. We will then issue a
   license for the device(s).

4. The license file location should be set in the environment variable `MYRTLE_LICENSE`.

   ```bash
   > export MYRTLE_LICENSE=<license file>
   ```

5. Check that the license for your device(s) is being recognised.

   ```bash
   > bin/vollo-tool license-check
   ```

      If successful, the output should look like this:

   ```
   Ok: found 2 devices with valid licenses
   ```
