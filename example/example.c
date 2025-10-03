#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */
#include "npy.h"
#include "utils.h"

#include <assert.h>
#include <ctype.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <vollo-rt.h>

#define NUM_RANDOM_INPUT_VECTORS 32

typedef struct ExampleOptions {
  // Path to program.vollo
  const char* program_path;
  // Index of the model to do inference with (defaults to 0)
  size_t model_index;
  // Number of inferences to compute
  size_t num_inferences;
  // Maximum number of jobs running concurrently (pipelined)
  size_t max_concurrent_jobs;
  // Number of extra inferences to run before starting to measure
  size_t num_warmup_inferences;
  // Time to wait in between each inference in nanoseconds (only for 1 concurrent job, defaults to
  // 0)
  size_t inference_spacing_ns;
  // Use the fp32 version of the API (compute is still done in bf16)
  bool fp32_api;
  // Use raw DMA buffers and skip input/output data copy
  bool raw_buffer_api;
  // Run the program in software simulation rather than on hardware
  bool run_in_vm;
  // The inputs are random numbers (as opposed to all 1.0), only when input_paths is not set
  bool random_input;
  // Output detailed measurements in JSON
  bool json;
  // Paths to input files (.npy format with float32 as dtype)
  // One path per model input
  const char* const* input_paths;
  // Paths to output files (.npy format with float32 as dtype)
  // One path per model output
  // Only the model outputs from the first inference are serialized to files even when
  // num_inferences > 1
  const char* const* output_paths;
} ExampleOptions;

// A small example of the Vollo API
//
// The steps are:
//
// - Init
// - Add accelerators
// - Load program
// - Add jobs
// - Poll the jobs until completion
// - Teardown/Cleanup
//
// This example is set up to get some timing data from the multiple runs of
// multiple concurrent jobs on a single model
static void vollo_example(ExampleOptions options) {
  struct timespec start_setup_time, start_warmup_time, start_compute_time, end_time,
    wait_time_start, wait_time_current;

  clock_gettime(CLOCK_MONOTONIC, &start_setup_time);

  // Print vollo-rt version
  const char* version = vollo_rt_version();
  fprintf(stderr, "Using vollo-rt version: %s\n", version);

  //////////////////////////////////////////////////
  // Init
  vollo_rt_context_t ctx;
  EXIT_ON_ERROR(vollo_rt_init(&ctx));

  //////////////////////////////////////////////////
  // Add accelerators
  size_t accelerator_index = 0;

  if (options.run_in_vm) {
    bool bit_accurate = true;
    EXIT_ON_ERROR(vollo_rt_add_vm(ctx, accelerator_index, bit_accurate));
  } else {
    EXIT_ON_ERROR(vollo_rt_add_accelerator(ctx, accelerator_index));
    fprintf(
      stderr,
      "Using Vollo accelerator with %ld core(s) and block_size %ld\n",
      vollo_rt_accelerator_num_cores(ctx, accelerator_index),
      vollo_rt_accelerator_block_size(ctx, accelerator_index));
  }

  //////////////////////////////////////////////////
  // Load program
  EXIT_ON_ERROR(vollo_rt_load_program(ctx, options.program_path));

  // The VM's hardware config is determined by the program
  if (options.run_in_vm) {
    fprintf(
      stderr,
      "Using Vollo VM with %ld core(s) and block_size %ld\n",
      vollo_rt_accelerator_num_cores(ctx, accelerator_index),
      vollo_rt_accelerator_block_size(ctx, accelerator_index));
  }

  //////////////////////////////////////////////////
  // Get model metadata
  size_t num_models = vollo_rt_num_models(ctx);

  size_t model_index = options.model_index;
  assert(model_index < num_models);

  size_t model_num_inputs = vollo_rt_model_num_inputs(ctx, model_index);
  size_t model_num_outputs = vollo_rt_model_num_outputs(ctx, model_index);

  const char* model_name = vollo_rt_model_name(ctx, model_index);

  fprintf(stderr, "Program metadata for model ");
  if (model_name != NULL) {
    fprintf(stderr, "%s (model index %ld):\n", model_name, model_index);
  } else {
    fprintf(stderr, "%ld:\n", model_index);
  }
  fprintf(stderr, "  %ld input(s) with shape(s): [", model_num_inputs);

  for (size_t model_input_ix = 0; model_input_ix < model_num_inputs; model_input_ix++) {
    const size_t* input_shape = vollo_rt_model_input_shape(ctx, model_index, model_input_ix);
    size_t input_shape_len = vollo_rt_model_input_shape_len(ctx, model_index, model_input_ix);

    for (size_t input_shape_ix = 0; input_shape_ix < input_shape_len; input_shape_ix++) {
      fprintf(stderr, "%ld", *input_shape);
      input_shape++;
      if (input_shape_ix != input_shape_len) {
        fprintf(stderr, ", ");
      }
    }
    fprintf(stderr, "]");

    if (model_input_ix + 1 < model_num_inputs) {
      fprintf(stderr, ", [");
    } else {
      fprintf(stderr, "\n");
    }
  }

  fprintf(stderr, "  %ld output(s) with shape(s): [", model_num_outputs);

  for (size_t model_output_ix = 0; model_output_ix < model_num_outputs; model_output_ix++) {
    const size_t* output_shape = vollo_rt_model_output_shape(ctx, model_index, model_output_ix);
    size_t output_shape_len = vollo_rt_model_output_shape_len(ctx, model_index, model_output_ix);

    for (size_t output_shape_ix = 0; output_shape_ix < output_shape_len; output_shape_ix++) {
      fprintf(stderr, "%ld", *output_shape);
      output_shape++;
      if (output_shape_ix != output_shape_len) {
        fprintf(stderr, ", ");
      }
    }
    fprintf(stderr, "]");

    if (model_output_ix + 1 < model_num_outputs) {
      fprintf(stderr, ", [");
    } else {
      fprintf(stderr, "\n");
    }
  }

  if (vollo_rt_model_input_streaming_dim(ctx, model_index, 0) >= 0) {
    fprintf(stderr, "  The model is streaming\n");
  }

  //////////////////////////////////////////////////
  // Setup input/output files

  NpyArray* input_arrays = (NpyArray*)malloc(sizeof(NpyArray) * model_num_inputs);

  if (options.input_paths[0] != NULL) {
    for (size_t i = 0; i < model_num_inputs; i++) {
      assert(options.input_paths[i] != NULL);

      input_arrays[i] = read_npy(options.input_paths[i]);

      // Check that the input has the number of input elements that the model expects
      assert(input_arrays[i].buffer_len == vollo_rt_model_input_num_elements(ctx, model_index, i));
      const size_t* input_shape = vollo_rt_model_input_shape(ctx, model_index, i);

      // Check that the input has the shape of input that the model expects
      for (size_t j = 0; j < input_arrays[i].shape_len; j++) {
        assert(input_arrays[i].shape[j] == (size_t)*input_shape);
        input_shape++;
      }
    }
  }

  NpyArray* output_arrays = (NpyArray*)malloc(sizeof(NpyArray) * model_num_outputs);

  if (options.output_paths[0] != NULL) {
    for (size_t i = 0; i < model_num_outputs; i++) {
      assert(options.output_paths[i] != NULL);

      output_arrays[i].buffer_len = vollo_rt_model_output_num_elements(ctx, model_index, i);
      output_arrays[i].buffer = (float*)malloc(sizeof(float) * output_arrays[i].buffer_len);
      {
        const size_t* output_shape = vollo_rt_model_output_shape(ctx, model_index, i);
        const uint8_t output_shape_len
          = (uint8_t)vollo_rt_model_output_shape_len(ctx, model_index, i);

        output_arrays[i].shape_len = output_shape_len;
        for (uint8_t output_shape_ix = 0; output_shape_ix < output_shape_len; output_shape_ix++) {
          output_arrays[i].shape[output_shape_ix] = (size_t)*output_shape;
          output_shape++;
        }
      }
    }
  }

  //////////////////////////////////////////////////
  // Setup inputs/outputs buffers

  // Number of input vectors
  // When random input is used, we randomly select a vector of random data for each inference
  size_t num_test_inputs = options.random_input ? NUM_RANDOM_INPUT_VECTORS : 1;
  bf16*** test_inputs = (bf16***)malloc(sizeof(bf16**) * num_test_inputs);
  float*** test_inputs_fp32 = (float***)malloc(sizeof(float**) * num_test_inputs);

  for (size_t i = 0; i < num_test_inputs; i++) {
    test_inputs[i] = (bf16**)malloc(sizeof(bf16*) * model_num_inputs);
    test_inputs_fp32[i] = (float**)malloc(sizeof(float*) * model_num_inputs);

    for (size_t j = 0; j < model_num_inputs; j++) {
      size_t num_input_elems = vollo_rt_model_input_num_elements(ctx, model_index, j);

      test_inputs[i][j] = options.raw_buffer_api ? vollo_rt_get_raw_buffer(ctx, num_input_elems)
                                                 : (bf16*)malloc(sizeof(bf16) * num_input_elems);
      test_inputs_fp32[i][j] = (float*)malloc(sizeof(float) * num_input_elems);

      for (size_t k = 0; k < num_input_elems; k++) {
        if (options.input_paths[0] != NULL) {
          test_inputs[i][j][k] = float_to_bf16(input_arrays[j].buffer[k]);
          test_inputs_fp32[i][j][k] = input_arrays[j].buffer[k];
        } else {
          test_inputs[i][j][k] = options.random_input ? rand_bf16() : 0x3f80;  // 1.0 as a bf16
          test_inputs_fp32[i][j][k] = options.random_input ? rand_float() : 1.0f;
        }
      }
    }
  }

  bf16** model_outputs = (bf16**)malloc(sizeof(bf16*) * model_num_outputs);
  float** model_outputs_fp32 = (float**)malloc(sizeof(float*) * model_num_outputs);

  for (size_t i = 0; i < model_num_outputs; i++) {
    size_t num_output_elems = vollo_rt_model_output_num_elements(ctx, model_index, i);

    model_outputs[i] = options.raw_buffer_api ? vollo_rt_get_raw_buffer(ctx, num_output_elems)
                                              : (bf16*)malloc(sizeof(bf16) * num_output_elems);
    model_outputs_fp32[i] = (float*)malloc(sizeof(float) * num_output_elems);
  }

  struct timespec* start_times
    = (struct timespec*)malloc(sizeof(struct timespec) * options.num_inferences);
  double* latencies = (double*)malloc(sizeof(double) * options.num_inferences);

  //////////////////////////////////////////////////
  // Run

  fprintf(stderr, "Starting %ld inferences\n", options.num_inferences);

  size_t total_inferences = options.num_warmup_inferences + options.num_inferences;
  size_t outstanding_jobs = 0;
  size_t inf_started = 0;
  size_t inf_completed = 0;

  // We don't need a user context here
  // Since we're only using 1 model, the inferences are guaranteed to complete
  // in the same order they were started
  uint64_t user_ctx = 0;

  clock_gettime(CLOCK_MONOTONIC, &start_warmup_time);

  while (inf_completed < total_inferences) {
    //////////////////////////////////////////////////
    // Add jobs

    while (outstanding_jobs < options.max_concurrent_jobs && inf_started < total_inferences) {
      size_t inf_ix = 0;

      // if it is not warmup
      if (inf_started >= options.num_warmup_inferences) {
        inf_ix = inf_started - options.num_warmup_inferences;
        clock_gettime(CLOCK_MONOTONIC, &start_times[inf_ix]);
      }

      int input_ix = options.random_input ? rand() % NUM_RANDOM_INPUT_VECTORS : 0;

      if (options.fp32_api) {
        EXIT_ON_ERROR(vollo_rt_add_job_fp32(
          ctx,
          model_index,
          user_ctx,
          (const float* const*)test_inputs_fp32[input_ix],
          model_outputs_fp32));
      } else {
        EXIT_ON_ERROR(vollo_rt_add_job_bf16(
          ctx, model_index, user_ctx, (const bf16* const*)test_inputs[input_ix], model_outputs));
      }

      inf_started++;
      outstanding_jobs++;
    }

    //////////////////////////////////////////////////
    // Poll the jobs

    size_t num_completed = 0;
    const uint64_t* completed_buffer = NULL;

    EXIT_ON_ERROR(vollo_rt_poll(ctx, &num_completed, &completed_buffer));

    if (num_completed > 0) {
      outstanding_jobs -= num_completed;

      struct timespec job_completed_time;
      clock_gettime(CLOCK_MONOTONIC, &job_completed_time);

      for (size_t i = 0; i < num_completed; i++) {
        // Serialize model outputs from first inference if output paths are set
        if (inf_completed == 0 && options.output_paths[0] != NULL) {
          for (size_t j = 0; j < model_num_outputs; j++) {
            assert(options.output_paths[j] != NULL);

            size_t num_output_elems = vollo_rt_model_output_num_elements(ctx, model_index, j);

            for (size_t k = 0; k < num_output_elems; k++) {
              if (options.fp32_api) {
                output_arrays[j].buffer[k] = model_outputs_fp32[j][k];
              } else {
                output_arrays[j].buffer[k] = bf16_to_float(model_outputs[j][k]);
              }
            }

            write_npy(options.output_paths[j], output_arrays[j]);
          }
        }

        // if it is not warmup
        if (inf_completed >= options.num_warmup_inferences) {
          size_t ix = inf_completed - options.num_warmup_inferences;

          latencies[ix] = diff_timespec_ns(start_times[ix], job_completed_time);
        }

        inf_completed++;
      }

      // Wait in between inferences
      if (options.inference_spacing_ns > 0) {
        clock_gettime(CLOCK_MONOTONIC, &wait_time_start);

        do {
          clock_gettime(CLOCK_MONOTONIC, &wait_time_current);
        } while (diff_timespec_ns_ll(wait_time_start, wait_time_current)
                 < (long long)options.inference_spacing_ns);
      }
    }
  }

  start_compute_time = start_times[0];
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  //////////////////////////////////////////////////
  // Summarize latencies

  latency_summary summary = summarize_latencies(options.num_inferences, latencies);

  double setup_time = diff_timespec_ns(start_setup_time, start_warmup_time) / NANOSECONDS;
  double warmup_time = diff_timespec_ns(start_warmup_time, start_compute_time) / NANOSECONDS;
  double compute_time = diff_timespec_ns(start_compute_time, end_time) / NANOSECONDS;
  double throughput = (double)options.num_inferences / compute_time;

  fprintf(stderr, "Done\n");

  if (options.json) {
    printf("{\n");
    printf("  \"options\": {\n");
    printf("    \"max_concurrent_jobs\": %ld,\n", options.max_concurrent_jobs);
    printf("    \"num_inferences\": %ld,\n", options.num_inferences);
    printf("    \"raw_buffer_api\": %d,\n", options.raw_buffer_api);
    printf("    \"inference_spacing_ns\": %ld\n", options.inference_spacing_ns);
    printf("  },\n");
    printf("  \"metrics\": {\n");
    printf("    \"time\": {\n");
    printf("      \"setup\": %f,\n", setup_time);
    printf("      \"warmup\": %f,\n", warmup_time);
    printf("      \"compute\": %f\n", compute_time);
    printf("    },\n");
    printf("    \"throughput\": %f,\n", throughput);
    printf("    \"latency_us\": {\n");
    printf("      \"mean\": %f,\n", summary.mean_latency_ns / 1000);
    printf("      \"median\": %f,\n", summary.median_latency_ns / 1000);
    printf("      \"p99\": %f,\n", summary.p99_latency_ns / 1000);
    printf("      \"worst\": %f\n", summary.worst_latency_ns / 1000);
    printf("    }\n");
    printf("  }\n");
    printf("}\n");
  } else {
    printf("Ran %ld inferences in %f s with:\n", options.num_inferences, compute_time);
    printf("  mean latency of %f us\n", summary.mean_latency_ns / 1000);
    printf("  99%% latency of %f us\n", summary.p99_latency_ns / 1000);
    printf("  throughput of %f inf/s\n", throughput);
  }

  //////////////////////////////////////////////////
  // Teardown/Cleanup

  free(latencies);
  free(start_times);

  for (size_t i = 0; i < model_num_outputs; i++) {
    if (!options.raw_buffer_api) {
      free(model_outputs[i]);
    }
    free(model_outputs_fp32[i]);
  }
  free(model_outputs);
  free(model_outputs_fp32);

  for (size_t i = 0; i < num_test_inputs; i++) {
    for (size_t j = 0; j < model_num_inputs; j++) {
      if (!options.raw_buffer_api) {
        free(test_inputs[i][j]);
      }
      free(test_inputs_fp32[i][j]);
    }
    free(test_inputs[i]);
    free(test_inputs_fp32[i]);
  }
  free(test_inputs);
  free(test_inputs_fp32);

  vollo_rt_destroy(ctx);

  if (options.input_paths[0] != NULL) {
    for (size_t i = 0; i < model_num_inputs; i++) {
      free_npy(input_arrays[i]);
    }
  }
  free(input_arrays);

  if (options.output_paths[0] != NULL) {
    for (size_t i = 0; i < model_num_outputs; i++) {
      free_npy(output_arrays[i]);
    }
  }
  free(output_arrays);
}

void print_help(const char* example_program) {
  printf(
    "\n"
    "USAGE:\n    %s [OPTIONS] <VOLLO_PROGRAM>\n\n"
    "ARGS:\n"
    "    <VOLLO_PROGRAM>\n"
    "        Path to the Vollo program file (e.g: ./program.vollo)\n"
    "\n"

    "OPTIONS:\n"
    "    -m, --model-index\n"
    "        Index of the model to do inference with\n"
    "        Defaults to 0\n"
    "\n"

    "    -i, --num-inferences\n"
    "        Number of inferences to compute\n"
    "        Defaults to 10_000\n"
    "\n"

    "    -F, --fp32-api\n"
    "        Use the fp32 version of the API (compute is still done in bf16)\n"
    "\n"

    "    -R, --raw-buffer-api\n"
    "        Use raw DMA buffers and skip input/output data copy\n"
    "\n"

    "    -v, --run-in-vm\n"
    "        Run the program in a VM instead of on an accelerator\n"
    "\n"

    "    -r, --random\n"
    "        Use random inputs instead of the constant 1.0\n"
    "\n"

    "    -c, --max-concurrent-jobs\n"
    "        Maximum number of jobs/inferences running concurrently (pipelined)\n"
    "        Defaults to 1\n"
    "\n"

    "    -w, --num-warmup-inferences\n"
    "        Number of extra inferences to run before starting to measure\n"
    "        Defaults to 10000\n"
    "\n"

    "    -s, --inference-spacing-ns\n"
    "        Time to wait in between each inference in nanoseconds (only for 1 concurrent job)\n"
    "        Defaults to 0\n"
    "\n"

    "    -j, --json\n"
    "        Output detailed measurements in JSON\n"
    "\n"

    "    -f, --input\n"
    "        Path to an input file (.npy format with bfloat16 as dtype)\n"
    "        Use this argument once per model input, in order of model input\n"
    "\n"

    "    -o, --output\n"
    "        Path to an output file (.npy format with bfloat16 as dtype)\n"
    "        Use this argument once per model output, in order of model output\n"
    "        Note: Only the model outputs from the first inference are serialized\n"
    "        to files even when num_inferences > 1\n"
    "\n"

    "    -h, --help\n"
    "        Prints this help information\n"
    "\n",
    example_program);
}

#define MAX_INPUT_PATH_COUNT 10
#define MAX_OUTPUT_PATH_COUNT 10

int main(int argc, char** argv) {
  const char* input_paths[MAX_INPUT_PATH_COUNT] = {NULL};
  size_t input_paths_count = 0;
  const char* output_paths[MAX_OUTPUT_PATH_COUNT] = {NULL};
  size_t output_paths_count = 0;

  ExampleOptions options;
  options.program_path = "";
  options.model_index = 0;
  options.fp32_api = false;
  options.raw_buffer_api = false;
  options.random_input = false;
  options.run_in_vm = false;
  options.max_concurrent_jobs = 1;
  options.num_inferences = 10000;
  options.num_warmup_inferences = 10000;
  options.inference_spacing_ns = 0;
  options.json = false;
  options.input_paths = input_paths;
  options.output_paths = output_paths;

  // Seed the random number generator
  srand((uint32_t)time(NULL));

  // Parse example options
  static struct option long_options[] = {
    {"model-index", required_argument, 0, 'm'},
    {"num-inferences", required_argument, 0, 'i'},
    {"fp32-api", no_argument, 0, 'F'},
    {"raw-buffer-api", no_argument, 0, 'R'},
    {"run-in-vm", no_argument, 0, 'v'},
    {"random", no_argument, 0, 'r'},
    {"max-concurrent-jobs", required_argument, 0, 'c'},
    {"num-warmup-inferences", required_argument, 0, 'w'},
    {"inference-spacing-ns", required_argument, 0, 's'},
    {"json", no_argument, 0, 'j'},
    {"input", required_argument, 0, 'f'},
    {"output", required_argument, 0, 'o'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0},
  };

  int opt = 0;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, "m:i:FRvrc:w:jf:o:h", long_options, &long_index)) != -1) {
    switch (opt) {
    case 'm': options.model_index = (size_t)strtoul(optarg, NULL, 10); break;
    case 'i': options.num_inferences = (size_t)strtoul(optarg, NULL, 10); break;
    case 'F': options.fp32_api = true; break;
    case 'R': options.raw_buffer_api = true; break;
    case 'v': options.run_in_vm = true; break;
    case 'r': options.random_input = true; break;
    case 'c': options.max_concurrent_jobs = (size_t)strtoul(optarg, NULL, 10); break;
    case 'w': options.num_warmup_inferences = (size_t)strtoul(optarg, NULL, 10); break;
    case 's': options.inference_spacing_ns = (size_t)strtoul(optarg, NULL, 10); break;
    case 'j': options.json = true; break;
    case 'f':
      input_paths[input_paths_count] = optarg;
      input_paths_count++;
      // We still need a NULL at the end of the array in case we don't specify enough inputs on the
      // CLI for the model
      assert(input_paths_count < MAX_INPUT_PATH_COUNT);
      break;
    case 'o':
      output_paths[output_paths_count] = optarg;
      output_paths_count++;
      // We still need a NULL at the end of the array in case we don't specify enough outputs on the
      // CLI for the model
      assert(output_paths_count < MAX_OUTPUT_PATH_COUNT);
      break;
    default: print_help(argv[0]); exit(opt == 'h' ? EXIT_SUCCESS : EXIT_FAILURE);
    }
  }

  if (optind == (argc - 1)) {
    options.program_path = argv[optind];
    fprintf(stderr, "Using program: \"%s\"\n", options.program_path);
  } else {
    print_help(argv[0]);
    exit(EXIT_FAILURE);
  }

  assert(options.max_concurrent_jobs > 0);
  assert(options.num_inferences > 0);

  if (options.random_input && options.input_paths[0] != NULL) {
    fprintf(stderr, "Options -r,--random and -f,--input are not compatible\n");
    exit(EXIT_FAILURE);
  }

  if (options.fp32_api && options.raw_buffer_api) {
    fprintf(stderr, "Options -F,--fp32-api and -R,--raw-buffer-api are not compatible\n");
    exit(EXIT_FAILURE);
  }

  if (options.raw_buffer_api && (options.max_concurrent_jobs > 1)) {
    fprintf(
      stderr,
      "Combination of -R,--raw-buffer-api and -c,--max-concurrent-jobs > 1 is not supported in "
      "this simple example\n");
    fprintf(stderr, "NOTE: output raw buffers cannot be reused for concurrent inferences\n");
    exit(EXIT_FAILURE);
  }

  if ((options.inference_spacing_ns > 0) && (options.max_concurrent_jobs > 1)) {
    fprintf(
      stderr,
      "Combination of -s,--inference-spacing-ns and -c,--max-concurrent-jobs > 1 is not supported "
      "in "
      "this simple example\n");
    exit(EXIT_FAILURE);
  }

  vollo_example(options);
}
