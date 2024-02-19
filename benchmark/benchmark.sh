#! /usr/bin/env bash
#
# Benchmark script of example Vollo models
set -e

info() {
  >&2 echo -e "\e[33m$1\e[0m"
}

if [ -z "$VOLLO_SDK" ]; then
  echo "VOLLO_SDK is not set - have you run 'source setup.sh'?"
  exit 1
fi

info "creating work directory"
mkdir -p work
cd work

if [ ! -d "vollo-venv" ]; then
  info "creating vollo-venv virtual environment"
  python3 -m venv vollo-venv
fi

info "activating vollo-venv virtual environment"
# Directory is created dynamically
# shellcheck disable=SC1091
source vollo-venv/bin/activate

info "installing Vollo Python libraries requirements"
pip3 install --upgrade pip
pip3 install "$VOLLO_SDK"/python/vollo_python-*.whl
pip3 install "$VOLLO_SDK"/python/vollo_torch-*.whl  --extra-index-url https://download.pytorch.org/whl/cpu

info "building example application"
cp -r "$VOLLO_SDK"/example example
chmod +w example
( cd example; make )

cmd="example/vollo-example"

info "Getting bitstream info for vollo device"
if "$VOLLO_SDK"/bin/vollo-tool bitstream-check "$VOLLO_SDK"/bitstream/vollo-ia840f.json &> /dev/null; then
  device="ia_840f"
  echo "Detected vollo-ia840f"
elif "$VOLLO_SDK"/bin/vollo-tool bitstream-check "$VOLLO_SDK"/bitstream/vollo-ia420f.json &> /dev/null; then
  device="ia_420f"
  echo "Detected vollo-ia420f"
else
  echo "Failed to find vollo device(s) matching this vollo-sdk"
fi

models=("identity-128" "mlp" "cnn")

info "Generating models"
for m in "${models[@]}"; do
  python3 "$VOLLO_SDK"/example/programs.py -m "$m" -c "$device"
done
echo
echo

echo "------------------------------------------------------------------"
echo "-- RUNNING INFERENCE BENCHMARKS ----------------------------------"
echo "------------------------------------------------------------------"
echo

for m in "${models[@]}"; do
    info "Running inference for $m:"
    $cmd "$m.vollo"
  echo
done
