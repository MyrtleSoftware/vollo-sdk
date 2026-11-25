## GPU input example (experimental)

This example writes Vollo inputs from an NVIDIA GPU. A CUDA kernel stores each job's input
straight into Vollo's PCIe input aperture, with no copy through host memory, and the outputs
arrive in host memory as usual. It checks every GPU job against an ordinary job given the same
input from the CPU.

It uses the experimental external-input API, described in the "External input" section of the
C API guide (`docs/mdbook/src/c-api.md`).

### Requirements

- An NVIDIA GPU and the CUDA toolkit (`nvcc`).
- A Vollo bitstream whose input aperture accepts writes wider than 64 bytes. The example prints
  the aperture's write limits and stops if they are too small.
- A PCIe platform that routes peer writes from the GPU to the Vollo card. Check this with your
  platform's PCIe topology, ACS and IOMMU configuration.
- A GPU Max Payload Size no larger than the aperture's largest write, which the example prints.
  Under different root ports the two devices can have different settings; Linux's
  `pci=pcie_bus_peer2peer` boot option sets every device to 128 bytes.

### Letting the GPU driver register the aperture

NVIDIA drivers register device memory for GPU access only for processes with `CAP_SYS_ADMIN`.
Run as any other user, the example's `cudaHostRegister` call fails with
`cudaErrorNotPermitted`. To let unprivileged applications register the aperture, set the
driver's `PeerMappingOverride` registry key, rebuild the initramfs so that the key also applies
at boot, and reload the driver:

```sh
echo 'options nvidia NVreg_RegistryDwords="PeerMappingOverride=1"' |
    sudo tee /etc/modprobe.d/nvidia-peermapping.conf
# Ubuntu, among others, loads the driver from the initramfs at boot, with the initramfs's own
# copy of /etc/modprobe.d.
sudo update-initramfs -u  # Debian and Ubuntu; sudo dracut -f on Fedora and RHEL
# Reloading needs a GPU with no users, including a display server.
sudo systemctl stop nvidia-persistenced  # if it is running
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe -a nvidia nvidia_modeset nvidia_drm nvidia_uvm
sudo systemctl start nvidia-persistenced # if it was running
grep RegistryDwords /proc/driver/nvidia/params  # shows PeerMappingOverride=1
```

If another file already sets `NVreg_RegistryDwords`, add the key to that value instead,
separated by a semicolon. The setting applies to every user of the GPU on the host. The driver
registers only ranges that are already mapped as device memory in the calling process, so a
process can direct GPU writes only at device memory it has mapped itself, such as the Vollo
aperture it opened.

### Building and running

Make sure the `VOLLO_SDK` environment variable is set by sourcing setup.sh from the Vollo SDK.

```bash
source setup.sh
export MYRTLE_LICENSE=<your-license-file>
cd example/gpu
make
./vollo-gpu-input
```

The Makefile builds machine code for every major GPU architecture, so the driver never compiles
the kernel's PTX itself, which fails ("the provided PTX was compiled with an unsupported
toolchain") when the CUDA toolkit is newer than the driver. `make CUDA_ARCH=native` builds for
this machine's GPUs only.

By default it runs 100 jobs of the identity program for the accelerator's block size from the
parent directory. Give another program as an argument; it must be stateless, since each GPU job
is compared with a reference job of the same input. To validate a platform, use the identity
program, whose outputs show every input element. `--device` selects the Vollo accelerator,
`--cuda-device` the GPU and `--num-jobs` the number of jobs.

### Writing your own GPU producer

The aperture is a device input port, not memory. Each input must be written exactly once,
complete with its zero padding, as PCIe writes that follow the rules in
`vollo_rt_get_external_input_aperture`'s documentation. Each 16-byte lane of a 64-byte input
word may arrive in any order, but exactly once. Never read the aperture.

Write each 64-byte word whole with one instruction, for example as 16-byte stores from four
adjacent threads of a warp, not as separate partial-word stores. A GPU may merge stores into one
write, and a store that leaves a word partly written can then be merged with the next word's
store into a write that starts inside a word and crosses into the next, which the card refuses.
The example's kernel stores aligned 16-byte vectors from consecutive threads, so each warp store
writes whole words. Since Volta, the threads of a warp can diverge and then store separately, so
compute every thread's value without branching on the thread, and call `__syncwarp()` just before
the store. Make the store volatile (the example uses `st.volatile` in inline PTX), so the compiler
keeps it as one 16-byte store rather than splitting it or merging it with others.

CUDA does not specify the PCIe writes a store produces, so validate any writer on your own GPU
and platform before using it for inference.
