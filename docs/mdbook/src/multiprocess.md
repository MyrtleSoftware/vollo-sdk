# Configuring the Accelerator for Multiple Processes (SR-IOV)

By default, `vollo-rt` uses the accelerator through its PCIe physical function (PF), bound to
`vfio-pci`. Only one process can open a `vfio-pci` device at a time, and a `vollo-rt` context can
only be driven from one thread, so only one thread can use the accelerator.

The V80, V80LL and Artena bitstreams support SR-IOV (single-root I/O virtualisation). This lets you
split a card into virtual functions (VFs). Each VF is a separate PCIe device with its own `vollo-rt`
context, so several threads of one process, several processes, or several virtual machines can each
drive a VF of their own at the same time. All of the VFs run one program, loaded onto the card's
cores, and their jobs share those cores. Agilex cards do not support VFs.

| Card       | Maximum VFs |
| ---------- | ----------- |
| V80, V80LL | 16          |
| Artena     | 8           |

Setting this up takes three steps:

1. [Create the VFs](#creating-the-vfs) with `load-kernel-driver.sh`. This needs root.
2. [Load a program onto the VFs](#loading-a-program-onto-the-vfs) with `vollo-tool setup-vfs`.
   This loads your program onto the card and writes one virtual program file for each VF. It does
   not need root.
3. [Use a VF](#using-a-vf) from each thread or process: open the VF and load its virtual program,
   in place of opening the card and loading the `.vollo` program.

The VFs do not survive a reboot, so after every reboot do steps 1 and 2 again. To set up the same
deployment as before, so that the applications can keep their virtual programs, see [Repeating a
deployment](#repeating-a-deployment).

## Requirements

In addition to the requirements for using the whole card:

- The card must be a V80, V80LL or Artena.
- SR-IOV support should be turned on in the BIOS. See [BIOS
  Requirements](system-requirements.md#bios-requirements).
- The host must run Linux 4.18 or later built with SR-IOV (`CONFIG_PCI_IOV`), `pci-pf-stub`
  (`CONFIG_PCI_PF_STUB`) and `vfio-pci` (`CONFIG_VFIO_PCI`).
- The IOMMU must be turned on, and must put each VF in an IOMMU group of its own.
  `load-kernel-driver.sh` checks for this.
- The card must be [licensed](licensing.md). `setup-vfs` checks the licence when it loads the
  program, so set `MYRTLE_LICENSE` for it as you would for any other program. Applications that use
  a VF do not need the licence.

## Creating the VFs

First find the card's PCI address. `vollo-tool list` prints it:

```sh
$ vollo-tool list
DEVICE  PCI ADDRESS    KIND  DRIVER
0       01:00.0        PF    vfio-pci
```

To create 16 VFs on that card (8 on an Artena):

```sh
sudo $VOLLO_SDK/load-kernel-driver.sh vfio --bdf 01:00.0 --num-vfs 16
```

This binds the PF to `pci-pf-stub`, creates the VFs, and binds each VF to `vfio-pci`. `--bdf` must
name function 0 of the card. `--num-vfs` can be at most the number in the table above. The script
refuses to change anything while the card or any of its VFs is open, and names the processes that
have it open.

`vollo-tool list` now shows the card with its VFs:

```text
DEVICE  PCI ADDRESS    KIND  DRIVER
0       01:00.0        PF    pci-pf-stub  16 VFs
0.0     01:02.0        VF    vfio-pci
0.1     01:02.1        VF    vfio-pci
...
```

Each VF is named `<card>.<vf>`, counting from 0. For example, `0.3` is VF 3 of card 0. Other
`vollo-tool` commands and the runtime accept these names.

If the BIOS reserved no PCIe address space for the VFs, the script stops before changing anything.
To fix this, turn on SR-IOV support in the BIOS or, if there is free address space beside the card,
boot Linux with `pci=realloc`.

## Loading a program onto the VFs

Once the VFs exist, load your program and generate the virtual programs:

```sh
$ vollo-tool setup-vfs --pf 0 --program program.vollo --output virtual-programs
Virtual programs published at virtual-programs
  VF 0.0    01:02.0        virtual-programs/vf-000.vollo-vf
  VF 0.1    01:02.1        virtual-programs/vf-001.vollo-vf
  ...
```

`setup-vfs` resets the card's accelerator, loads the program and enables the VFs. Without a
placement file, every VF can run every model in the program, with the same model indices as in the
program. The VFs stay set up after `setup-vfs` exits.

Each application opens one VF and loads that VF's `.vollo-vf` file, so it needs to know both, for
example from its command line or its configuration. The file for VF `0.3` is `vf-003.vollo-vf`. See
[Using a VF](#using-a-vf).

- `--pf` is the card's device index or PCI address, as `vollo-tool list` shows.
- `--program` is the compiled `.vollo` program.
- `--output` is the directory for `setup-vfs` to write one `vf-NNN.vollo-vf` file for each enabled
  VF into, and a [`placement.toml`](#repeating-a-deployment) that records the deployment. It must
  either not exist yet or be the output of an earlier `setup-vfs`, which `setup-vfs` replaces as a
  whole once the card is set up, so that a reader sees either the old files or the new ones.
- `--num-vfs N` enables only the first N VFs: `--num-vfs 3` enables VFs 0, 1 and 2. N can be
  anything from 1 up to the number of VFs you created, and no more than the card can run jobs for.
  The other VFs still exist, but the card runs nothing for them and `setup-vfs` writes no virtual
  program for them. Without this option, every VF you created is enabled, up to the number the
  card can run jobs for.
- `--placement placement.toml` says which models each VF can run, in place of `--num-vfs`. See
  [Choosing models for each VF](#choosing-models-for-each-vf).
- `--cut-through`, `--share-stateful-models` and `--no-reset-state-on-load` change how the VFs run
  jobs. See [Deployment settings](#deployment-settings).

If the program has a stateful model, such as a streaming LSTM, read [Stateful
models](#stateful-models) first. By default `setup-vfs` refuses to put a stateful model on more
than one VF, so the command above fails unless only one VF is enabled, and even on one VF the model
must be compiled with `generate_state_reset = True`.

`setup-vfs` checks the program and placement before it changes anything on the card. It refuses to
run while any VF is open (see [Changing a deployment](#changing-a-deployment)).

## Using a VF

Each thread or process that drives a VF creates its own context, opens its VF with
`vollo_rt_add_device` and loads that VF's virtual program with `vollo_rt_load_program`, in place of
the usual `vollo_rt_add_accelerator` and `.vollo` program:

```c
vollo_rt_context_t ctx;
EXIT_ON_ERROR(vollo_rt_init(&ctx));
EXIT_ON_ERROR(vollo_rt_add_device(ctx, 0, "0.3"));
EXIT_ON_ERROR(vollo_rt_load_program(ctx, "virtual-programs/vf-003.vollo-vf"));
```

Here `0` is the accelerator's index within this context, and `"0.3"` is the VF to open. The VF's
PCI address, such as `"01:02.3"`, works too.

In Python:

```python
import vollo_rt

with vollo_rt.VolloRTContext() as ctx:
    ctx.add_device(0, "0.3")
    ctx.load_program("virtual-programs/vf-003.vollo-vf")
```

After that, you use `vollo-rt` exactly as you would with a whole card. See the [C API](c-api.md).
As with a whole card, each context must be driven from one thread; to drive the card from several
threads of one process, give each thread a context and a VF of its own.

The model indices are those of the VF, which are the program's own unless a
[placement](#choosing-models-for-each-vf) says otherwise.

The example application in the SDK runs on a VF too:

```sh
./example/vollo-example --device 0.3 virtual-programs/vf-003.vollo-vf
```

See `example/multi-process.sh` in the SDK for a fully worked example. It runs `vollo-tool setup-vfs`
with the placement in `example/example-placement.toml`, which enables four VFs. It then runs
`vollo-example` on the identity program from four processes at once.

When it loads a virtual program, the runtime checks that the file matches the card's current
deployment, and refuses a file from an earlier `setup-vfs` that set up something different. Always
use the files from the latest `setup-vfs`.

A context holds its VF from the moment it opens it until it is destroyed (`vollo_rt_destroy`).
Wait for its jobs to complete before destroying it. If a context is destroyed with jobs still
running, or its process exits or is killed, the kernel resets its VF when it is closed, which
discards those jobs; the other VFs carry on.

### Using a VF in a virtual machine

A VF can be assigned to a virtual machine like any other PCIe device. Give the VM the VF behind a
PCIe root port, so that the VM can reset it. A process in the VM opens the VF by the PCI address the
VM sees, such as `vollo_rt_add_device(ctx, 0, "00:05.0")`, since the VM has no PF and so no
`<card>.<vf>` names. Copy the VF's `.vollo-vf` file into the VM.

A VF assigned to a VM counts as in use only while a context in the VM has it open, so the VM can
keep its VF while you run `setup-vfs` again.

## Choosing models for each VF

A placement file is TOML. It says how many VFs to enable and which models each one can run:

```toml
schema = "vollo-vf-placement/v1"
# The number of VFs to enable, as with --num-vfs. VFs are numbered from 0, so
# 3 enables VFs 0, 1 and 2.
vf_count = 3

# Every VF appears in exactly one mapping.
[[mapping]]
# VF numbers, or "all".
vfs = [0, 1]
# Models by name or by index in the program, or "all".
models = ["fraud-detector"]

[[mapping]]
vfs = [2]
# The order sets the model indices the VF sees: VF 2 runs risk-score as model 0,
# and fraud-detector as model 1.
models = ["risk-score", "fraud-detector"]
```

Pass the file with `setup-vfs --placement placement.toml`. Models are named as they were when the
program was compiled, and a name must match exactly one model. A model can appear in several
mappings, with a different index in each.

A model can also be given as `{ index = 2, name = "risk-score" }`, as in the placement that
[`setup-vfs` writes](#repeating-a-deployment): this picks model 2 of the program, and `setup-vfs`
refuses the placement if that model is not named `risk-score`.

A placement can also set the [deployment settings](#deployment-settings). If you use a placement,
you must set them in the file, not on the command line.

## Deployment settings

These settings apply to every VF. Without a placement they are command-line options of
`setup-vfs`; with one, they are keys in the file.

| Option                     | Placement key           | Default |
| -------------------------- | ----------------------- | ------- |
| `--cut-through`            | `cut_through`           | false   |
| `--share-stateful-models`  | `share_stateful_models` | false   |
| `--no-reset-state-on-load` | `reset_state_on_load`   | true    |

### Cut-through

By default a job starts on the card only once all of its input has arrived there. With cut-through,
a job starts as soon as its request arrives, before all of its input has. This can reduce latency.
However, a job whose input stops arriving holds up every VF's jobs until its own VF is reset, as
happens when its context is destroyed or its process exits. Use cut-through only when every
application using a VF can be trusted to finish the input of every job it submits.

### Stateful models

A stateful model, such as a streaming model, keeps a single state on the card. Applications on VFs
that can both run it would share that state, and interleave their inferences through it. So `setup-vfs`
refuses to place a stateful model on more than one VF, including by default when every VF gets
every model, unless `share_stateful_models` is set. Set it only when the applications coordinate their
inferences. `Program.model_is_stateful` tells you whether a model is stateful.

When a context loads its virtual program, each stateful model on its VF starts from its initial
state, as when a program is loaded onto a whole card. For this, a stateful model placed on one VF
must be compiled with `generate_state_reset = True`, and `setup-vfs` refuses it otherwise.
`vollo_rt_add_reset_job` works on a VF as it does on a whole card.

With `reset_state_on_load = false` (`--no-reset-state-on-load`), loading a virtual program does
not reset any state: a stateful model starts from the state the VF's previous user left, and needs
no `generate_state_reset`.

A stateful model shared between VFs is never reset when a VF loads its program, and
`vollo_rt_add_reset_job` on it fails, since that would clear the state the other applications are
using.

## Limits

- A program used on VFs can have at most 256 models. A model compiled with
  `generate_state_reset = True` counts twice, so a program in which every model has a state reset
  can have at most 128. `setup-vfs` refuses a program past the limit.
- Partial updates (`vollo_rt_add_job_partial_update` and its bf16 form) work on a VF only when the
  program has a single model and a single VF is enabled. Elsewhere they fail with an error.

## Changing a deployment

To change the program, the models or the number of enabled VFs, run `setup-vfs` again, with the
same output directory to replace the files in it, or a new one. You do not need to run
`load-kernel-driver.sh` again unless you want more VFs than you created.

First stop everything that uses a VF: stop submitting jobs, poll until every submitted job has
completed, and destroy the contexts. Keep them stopped until `setup-vfs` succeeds, then give them
the new `.vollo-vf` files. While `setup-vfs` runs, opening a VF fails with `cannot claim <vf>: its
PF holds the card`.

`setup-vfs` refuses to start while any process has a VF open, and names the VFs and the processes
it can see holding them. `vollo-tool reset` on a card with VFs refuses in the same way.
`vollo-tool reset --force` resets the card anyway, discarding the work of every VF.

If you run `setup-vfs` again with the same program and the same deployment, the existing
`.vollo-vf` files keep working. If you change the program, the enabled VFs, the models a VF can run
or their order, or cut-through, the runtime refuses the old files. If you change only
`reset_state_on_load`, it does not: see [Repeating a deployment](#repeating-a-deployment).

## Repeating a deployment

Every `setup-vfs` writes a `placement.toml` beside the virtual programs. It records the deployment
in full: the number of enabled VFs, all three [deployment settings](#deployment-settings), a digest
of the `.vollo` program, and each VF's models by index and name. VFs with the same models share a
mapping. For the placement in [Choosing models for each VF](#choosing-models-for-each-vf), it reads
(with the digest shortened):

```toml
# Written by vollo-tool setup-vfs for the card at 0000:01:00.0. Pass it to setup-vfs --placement to set up the same deployment again, on this card or another.
schema = "vollo-vf-placement/v1"
vf_count = 3
cut_through = false
share_stateful_models = false
reset_state_on_load = true

[program]
digest = "sha256:9f2c…"

[[mapping]]
vfs = [0, 1]
models = [{ index = 0, name = "fraud-detector" }]

[[mapping]]
vfs = [2]
models = [{ index = 2, name = "risk-score" }, { index = 0, name = "fraud-detector" }]
```

Its main use is to set up the same deployment again, for example after a reboot. Run
`load-kernel-driver.sh` as before, then pass the file back with the same program:

```sh
vollo-tool setup-vfs --pf 0 --placement virtual-programs/placement.toml \
  --program program.vollo --output <dir>
```

Each `.vollo-vf` file carries a cookie computed from the program and the deployment: which VFs are
enabled, cut-through, and each VF's models. The same program and placement give the same cookie,
so the applications' existing `.vollo-vf` files keep working, including copies inside virtual
machines, and there is nothing to hand out again. `setup-vfs` refuses a different program, with
`the placement was resolved against program sha256:<digest>, but this program is
sha256:<digest>`.

To use the placement with a recompiled program, delete its `[program]` table. `setup-vfs` then
accepts any program, but still checks that each model's name matches its index, and fails with
`mapping <m>: program model <index> is "<new name>", not "<name>"` if the models have moved,
rather than give a VF the wrong model. The virtual programs are then new, so hand them out.

`reset_state_on_load` is recorded in the file but not in the cookie. If you change only that, the
old `.vollo-vf` files still load, with the old setting, so hand out the new ones.

## Checking a deployment

`vollo-tool list --vf-state` also shows which VFs are enabled, and which are open and by which
processes. While a command such as `setup-vfs` is working on the card, it shows the card as `busy`
and names that command, and shows its VFs' state as `unknown`.

`vollo-tool inspect-vfs` shows the card's deployment: how many VFs exist, which are enabled, which
are held and by whom, whether cut-through is on, and whether the PF holds the card. To check that a
virtual program matches the card, pass it with `--program`. To show which models each VF was given,
pass the `placement.toml` that `setup-vfs` wrote with `--placement`. While a command such as
`setup-vfs` is working on the card, `inspect-vfs` does not read it, and names that command
instead:

```sh
vollo-tool inspect-vfs --pf 0 \
  --program virtual-programs/vf-000.vollo-vf \
  --placement virtual-programs/placement.toml
```

`inspect-vfs` names only processes whose open files you are allowed to see, so run it as root to
see them all.

## Permissions and isolation

Only `load-kernel-driver.sh` needs root. So that you can run `setup-vfs` without root, it lets every
user on the host map the PF's registers, and it publishes a copy of the PF's PCIe configuration
under `/run/vollo/pci/`. With these default permissions, any user on the host can replace the
program, change which VFs are enabled, or reset the card. The script also lets every user open
every VF (`/dev/vfio/<group>` and the VF's files under `/sys/bus/pci/devices/`).

To give VFs to users who do not trust each other, restrict each VF's `/dev/vfio/<group>` node and
its files under `/sys/bus/pci/devices/<VF>` to the user it is for, and the PF's files under
`/sys/bus/pci/devices/<PF>` to whoever runs `setup-vfs`. When `load-kernel-driver.sh` changes the
number of VFs, the VFs that remain keep the owner and mode of their `/dev/vfio` node.

VFs are isolated from each other's faults: an application that misbehaves or dies on one VF cannot break
another VF, except as described under [cut-through](#cut-through). They are not isolated from each
other's monitoring: every VF can read the card's status for all VFs, including how many jobs each
VF has run and has outstanding, and the card's temperature. Do not rely on VFs to keep one user's
activity on the card private from another.

## Returning to using the whole card

You cannot use the PF in the usual way while VFs exist. To remove them, stop everything that is
using a VF, then run:

```sh
sudo $VOLLO_SDK/load-kernel-driver.sh vfio --bdf 01:00.0 --num-vfs 0
```

This removes the VFs and binds the PF to `vfio-pci` again. `load-kernel-driver.sh` without
`--num-vfs` leaves a card that has VFs alone, so this is the only way back short of a reboot.

## Troubleshooting

- **`setup-vfs` fails.** If it fails while checking the program and placement, it has not changed
  the card. If it fails after that, every VF is left disabled, and the previous deployment is not
  restored. Fix the cause and run `setup-vfs` again. If it stopped partway, for example because it
  was killed, run it again.
- **`setup-vfs` fails with `could not enable the VFs, nor gate them again`.** Whether each VF is
  usable is unknown, so `setup-vfs` keeps its output directory, and notes `the card stays held, so
  every tenant is refused until setup-vfs succeeds on it`. Fix the cause and run `setup-vfs`
  again.
- **Opening a VF fails with `cannot claim <vf>: its PF holds the card`.** A `vollo-tool` command
  holds the card while it works on it, so retry once it has finished. If none is running, one was
  killed while it held the card, for example `vollo-tool reset`, and `vollo-tool release-card --pf
  <card>`, which the message suggests, releases it. After a `setup-vfs` that stopped partway, run
  `setup-vfs` again instead.
- **`setup-vfs` fails with `cannot take <card>: VF <vf> is claimed`, but no process has the VF
  open.** The message notes `<vf> is claimed but not open: its last tenant stopped without
  releasing it, and the VF was not reset`. A context destroyed with jobs running, or whose process
  exited, holds its VF until the kernel resets the VF. That reset needs the VF's `reset_method`
  (`/sys/bus/pci/devices/<VF>/reset_method`) to start with `flr`, as it does by default.
  `setup-vfs` and the runtime warn when it does not, with `<vf> is not reset by FLR (reset_method:
  <methods>)`. If the kernel could not reset the VF, the VF stays held until it is next opened, or
  until root resets it with `echo 1 > /sys/bus/pci/devices/<VF>/reset`.
- **`load-kernel-driver.sh` fails with `the card is in use`.** Each note names a VF or the PF and
  the process that has it open, such as `VF 0000:01:02.0 is open by pid 4242 (my-app)`. Stop those
  processes and run the command again.
- **`load-kernel-driver.sh` fails partway.** Stop everything using the card and run the same
  command again.
- **Jobs on every VF stop completing.** With [cut-through](#cut-through), a job whose input stopped
  arriving holds up every VF. Destroy the context that submitted it, or stop its process, which
  resets its VF.
- **Loading a virtual program fails with `the VF's stateful models were not reset within 5s`.**
  The card has stopped running jobs, for example because of a stalled cut-through job. Check the
  card with `vollo-tool inspect-vfs`, then destroy the context and open the VF again: a second
  `vollo_rt_load_program` on the same context fails.
