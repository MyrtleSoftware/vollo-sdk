#!/usr/bin/env bash
# Run the identity program from multiple processes with vollo-example, each process using its
# own SR-IOV virtual function (VF) of one card. See "Configuring the Accelerator for Multiple
# Processes" in the Vollo documentation.
#
# Usage:
#   ./multi-process.sh [<card>]
#
#   <card>  the card's device index, as `vollo-tool list` shows it (default 0)
#
# Before running it:
#   - Source setup.sh from the Vollo SDK and set MYRTLE_LICENSE.
#   - Program the card with a V80, V80LL or artena Vollo bitstreams: these are
#     the bitstreams that support VFs.
#   - Create at least as many VFs on the card as example-placement.toml enables (four). This
#     needs root, and is needed again after a reboot:
#       sudo $VOLLO_SDK/load-kernel-driver.sh vfio --bdf <PF PCI address> --num-vfs 4
#     `vollo-tool list` then shows the card's VFs beneath it, named <card>.0, <card>.1, ...
#
# Nothing in this script needs root. It leaves the VFs in place; to use the card without VFs
# again, stop every process using a VF, then run
#   sudo $VOLLO_SDK/load-kernel-driver.sh vfio --bdf <PF PCI address> --num-vfs 0
set -euo pipefail

card=${1:-0}

if [[ -z ${VOLLO_SDK:-} ]]; then
  echo "VOLLO_SDK is not set: source setup.sh from the Vollo SDK first" >&2
  exit 1
fi

cd "$(dirname "$0")"
make vollo-example

# One process per VF that example-placement.toml enables
NUM_PROCESSES=$(sed -n 's/^vf_count *= *\([0-9]*\).*/\1/p' example-placement.toml)
if [[ -z $NUM_PROCESSES ]]; then
  echo "Could not read vf_count from example-placement.toml" >&2
  exit 1
fi

workdir=$(mktemp -d)
# The processes started in step 2. If the script stops early, for example on Ctrl-C, stop them
# too, so that they don't keep holding their VFs.
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true; wait; rm -rf "$workdir"' EXIT

##################################################
# 1. Set up the VFs (the card's operator does this, once per program).
#
# setup-vfs loads the program onto the card through its physical function (PF), enables the VFs
# that example-placement.toml asks for, gives each one the models example-placement.toml lists
# for it, and writes one virtual program per VF, vf-000.vollo-vf for VF 0 and so on.
echo "Setting up ${NUM_PROCESSES} VFs of card ${card} as example-placement.toml describes"
vollo-tool setup-vfs --pf "$card" --placement example-placement.toml \
  --program identity_b32.vollo --output "$workdir/virtual-programs"

##################################################
# 2. Run one process per VF, all at the same time.
#
# Each process is an ordinary vollo-rt application: vollo_rt_add_device opens its VF by name
# (VF <vf> of card <card> is "<card>.<vf>"), and vollo_rt_load_program loads that VF's virtual
# program in place of a .vollo program. The processes share the card while they run.
#
# A VF numbers its models from 0, in the order example-placement.toml lists them for the VF
# (rather than as the program does). example-placement.toml gives each VF only the identity
# model, so here we just use model index 0
#
# Each process runs a million inferences, about a second's worth, so that they overlap.
for ((vf = 0; vf < NUM_PROCESSES; vf++)); do
  program=$(printf '%s/virtual-programs/vf-%03d.vollo-vf' "$workdir" "$vf")
  echo "Starting a process on VF ${card}.${vf} with ${program##*/}"
  ./vollo-example --device "${card}.${vf}" --model-index 0 --num-inferences 1000000 --random \
    "$program" \
    > "$workdir/vf-${vf}.log" 2>&1 &
  pids+=("$!")
done

##################################################
# 3. Wait for every process, and show what each one printed.
failed=0
for ((vf = 0; vf < NUM_PROCESSES; vf++)); do
  log=$workdir/vf-${vf}.log
  status=0
  wait "${pids[$vf]}" || status=$?
  echo
  echo "=== Process on VF ${card}.${vf} ==="
  cat "$log"
  if ((status != 0)); then
    echo "FAILED: the process on VF ${card}.${vf} exited with status ${status}" >&2
    failed=1
  fi
done
# Every process has exited, so there is nothing left for the trap to stop.
pids=()

echo
if ((failed)); then
  echo "Some processes failed" >&2
  exit 1
fi
echo "All ${NUM_PROCESSES} processes ran the identity program, each on its own VF"
