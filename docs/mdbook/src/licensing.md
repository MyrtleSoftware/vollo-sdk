# Licensing

Vollo is licensed on a per-device basis.

## Redeeming licenses with vollo-tool

You will receive a `purchase-token` with your Vollo purchase. The `purchase-token` can be used to redeem Vollo licenses for a set number of devices.

To see the number of credits (i.e. the number of devices which can be redeemed) on your `purchase-token`, run:

```sh
bin/vollo-tool license num-remaining-devices -t <purchase-token>
```

To redeem devices on your purchase token:

1. Load the kernel driver if you haven't already done so:

   ```sh
   sudo ./load-kernel-driver.sh
   ```

2. Run `vollo-tool device-ids`. This will enumerate all Vollo accelerators and output their device IDs.

   ```sh
   bin/vollo-tool device-ids | tee vollo.devices
   ```

3. Run `vollo-tool license redeem-device`, passing the device IDs you wish to generate licenses for. This will print a breakdown of which devices will consume credits on the `purchase-token`.

   ```sh
   bin/vollo-tool license redeem-device -t <purchase-token> --device-ids <device IDs>
   ```

   Alternatively you can pass the `vollo.devices` output from the previous step if you wish to redeem licenses for all devices.

   ```sh
   bin/vollo-tool license redeem-device -t <purchase-token> --device-id-file <device ID file>
   ```

4. When you have confirmed which devices will consume credits on the `purchase-token`, run `vollo-tool license redeem-device --consume-credits` to generate the licenses.
   The licenses will be printed to `stdout`.

   ```sh
   bin/vollo-tool license redeem-device -t <purchase-token> --device-ids <device IDs> --consume-credits | tee vollo.lic
   ```

The licenses redeemed on a purchase token can be viewed at any time by running `vollo-tool license view-licenses`:

```sh
bin/vollo-tool license view-licenses -t <purchase-token> | tee vollo.lic
```

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
