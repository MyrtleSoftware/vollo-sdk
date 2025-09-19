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

if [ ! -f vollo-venv/bin/activate ]; then
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
mkdir -p example
cp "$VOLLO_SDK"/example/{*.{h,c},Makefile} example/
chmod +w example
( cd example; make vollo-example)

info "Getting hardware config for vollo device"
"$VOLLO_SDK"/bin/vollo-tool read-hw-config | jq '.[0].hw_config' > hw_config.json

echo "------------------------------------------------------------------"
echo "-- COMPILING AND RUNNING BENCHMARK PROGRAMS ----------------------"
echo "------------------------------------------------------------------"
echo

cmd="example/vollo-example"

for m in $(python3 "$VOLLO_SDK"/example/programs.py list-models); do
  info "Compiling program for $m example"
  python3 "$VOLLO_SDK"/example/programs.py compile-model -m "$m" -c hw_config.json 2>/dev/null || echo "Could not compile $m for config"
  echo
done

for spacing_ns in 0 20000; do
  if [[ $spacing_ns -gt 0 ]]; then
    spacing=spaced
  else
    spacing=unspaced
  fi

  for model_type in identity mlp cnn lstm; do
    metrics_csv="$model_type-$spacing-latency-metrics.csv"

    # Write csv header (consists of a model_type dependent description and latency metrics)
    for m in $(python3 "$VOLLO_SDK"/example/programs.py list-models --model-type $model_type); do
      python3 "$VOLLO_SDK"/example/programs.py compile-model --describe-only -m "$m" -c hw_config.json \
        | jq -r 'keys_unsorted | . + ["Mean latency (us)", "99th percentile latency (us)"] | join(",")' > $metrics_csv
      # Only need to do this once per model type
      break
    done

    # Write csv rows
    for m in $(python3 "$VOLLO_SDK"/example/programs.py list-models --model-type $model_type); do
      # Skip models that couldn't be compiled
      [[ ! -f "$m.vollo" ]] && continue

      model_description_json=$(python3 "$VOLLO_SDK"/example/programs.py compile-model --describe-only -m "$m" -c hw_config.json)
      model_description_row=$(echo "$model_description_json" | jq -r 'join(",")')

      latency_json=$($cmd --inference-spacing-ns $spacing_ns --json "$m.vollo" | tee /dev/stderr)
      latency_row=$(echo "$latency_json" | jq -r '.metrics.latency_us | [.mean, .p99] | join(",")')

      combined_row="$model_description_row,$latency_row"

      echo "$combined_row" >> $metrics_csv
    done
  done
done

# cleanup
for m in $(python3 "$VOLLO_SDK"/example/programs.py list-models); do
  if [[ -f "$m.vollo" ]]; then
    rm "$m.vollo"
  fi
done

if [[ -n "$RUN_IO_TEST" ]]; then
  echo "------------------------------------------------------------------"
  echo "-- COMPILING AND RUNNING IO ONLY TEST PROGRAMS -------------------"
  echo "------------------------------------------------------------------"
  echo

  echo "input values, output values, mean latency (us), 99th percentile latency (us)" > io-test-metrics.csv

  for input_size_pow in $(seq 5 1 13); do
    input_size=$((2 ** input_size_pow))
    for output_size_pow in $(seq 5 1 13); do
      output_size=$((2 ** output_size_pow))
      info "Compiling program for IO only test: input_size=$input_size, output_size=$output_size"
      m=$(python3 "$VOLLO_SDK"/example/io-test-programs.py -c hw_config.json "$input_size" "$output_size")

      info "Program compiled $m, now running inference"
      echo -n "$input_size,$output_size," >> io-test-metrics.csv
      $cmd --num-inferences=100000 --json "$m" | tee /dev/stderr | jq -r '.metrics.latency_us | [.mean, .p99] | join(",")' >> io-test-metrics.csv
      echo

      # cleanup
      rm "$m"
    done
  done
fi
