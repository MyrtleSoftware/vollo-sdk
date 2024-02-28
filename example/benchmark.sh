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
pip3 install "$VOLLO_SDK"/python/vollo_compiler-*.whl
pip3 install "$VOLLO_SDK"/python/vollo_torch-*.whl  --extra-index-url https://download.pytorch.org/whl/cpu

info "building example application"
cp -r "$VOLLO_SDK"/example example
chmod +w example
( cd example; make )

cmd="example/vollo-example"

info "Getting hardware config for vollo device"
"$VOLLO_SDK"/bin/vollo-tool read-hw-config | jq '.[0].hw_config' > hw_config.json

models=("identity-128" "mlp" "cnn")

info "Generating models"
for m in "${models[@]}"; do
  python3 "$VOLLO_SDK"/example/programs.py -m "$m" -c hw_config.json
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
