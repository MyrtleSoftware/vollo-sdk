# Licensing

Vollo is licensed on a per-device basis.

## Obtaining a license

To obtain a license for your device(s), you will need to provide Myrtle with the device ID(s).

1. Load the kernel driver if you haven't already done so:

   ```sh
   sudo ./load-kernel-driver.sh
   ```

2. Run `vollo-tool device-ids`, this will enumerate all Vollo accelerators and
   and output their device IDs.

   ```sh
   bin/vollo-tool device-ids | tee vollo.devices
   ```

3. Send the `vollo.devices` to `vollo-license@myrtle.ai`. We will then issue a
   license for the device(s).

## Installing a license

1. The license file location should be set in the environment variable `MYRTLE_LICENSE`.

   ```sh
   export MYRTLE_LICENSE=<license file>
   ```

2. Check that the license for your device(s) is being recognised.

   ```sh
   bin/vollo-tool license-check
   ```

   If successful, the output should look like this:

   ```output
   Ok: found 2 devices with valid licenses
   ```
