## c example

This simple example runs generated data on the given model and collects the results. It is a minimal
usage of the VOLLO API.

### Building

Make sure the `VOLLO_SDK` environment variable is set by sourcing setup.sh from the VOLLO SDK.

```bash
source setup.sh
export MYRTLE_LICENSE=<your-license-file>
cd example
make
./vollo-example <.vollo program>
```
