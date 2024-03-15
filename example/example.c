#include "npy.h"

#include <assert.h>
#include <ctype.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <vollo-rt.h>

// Helper to exit when an error is encountered
#define EXIT_ON_ERROR(expr)                 \
  do {                                      \
    vollo_rt_error_t _err = (expr);         \
    if (_err != NULL) {                     \
      fprintf(stderr, "error: %s\n", _err); \
      exit(EXIT_FAILURE);                   \
    }                                       \
  } while (0)

#define NUM_RANDOM_INPUT_VECTORS 1024
#define NANOSECONDS (1000 * 1000 * 1000)

static float rand_float();
static bf16 rand_bf16();
static float bf16_to_float(bf16 x);
static bf16 float_to_bf16(float x);
static int compare_double(const void* a, const void* b);
static double diff_timespec_ns(struct timespec from, struct timespec to);

typedef struct ExampleOptions {
  // Path to program.vollo
  const char* program_path;
  // Number of inferences to compute
  size_t num_inferences;
  // Maximum number of jobs running concurrently (pipelined)
  size_t max_concurrent_jobs;
  // Number of extra inferences to run before starting to measure
  size_t num_warmup_inferences;
  // Use the fp32 version of the API (compute is still done in bf16)
  bool fp32_api;
  // The input is random numbers (as opposed to all 1.0), only when input_path is not set
  bool random_input;
  // Output detailed measurements in JSON
  bool json;
  // Path to an input file (.npy format with float32 as dtype)
  const char* input_path;
  // Path to an output file (.npy format with float32 as dtype)
  // Only the first output is serialized to a file even when num_inferences > 1
  const char* output_path;
} ExampleOptions;

// A small example of the VOLLO API
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
  struct timespec start_setup_time, start_warmup_time, start_compute_time, end_time;

  clock_gettime(CLOCK_MONOTONIC, &start_setup_time);

  NpyArray input_array = {0};
  if (options.input_path != NULL) {
    input_array = read_npy(options.input_path);
  }

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
  EXIT_ON_ERROR(vollo_rt_add_accelerator(ctx, accelerator_index));

  //////////////////////////////////////////////////
  // Load program
  EXIT_ON_ERROR(vollo_rt_load_program(ctx, options.program_path));

  //////////////////////////////////////////////////
  // Get model metadata
  size_t num_models = vollo_rt_num_models(ctx);
  assert(num_models == 1);

  size_t model_index = 0;

  size_t model_num_inputs = vollo_rt_model_num_inputs(ctx, model_index);
  assert(model_num_inputs == 1);

  size_t model_num_outputs = vollo_rt_model_num_inputs(ctx, model_index);
  assert(model_num_outputs == 1);

  fprintf(stderr, "Program metadata:\n");
  fprintf(stderr, "  %ld input with shape: [", model_num_inputs);

  // Initialised to 1 to get the product of the shape dims
  size_t num_input_elems = 1;
  {
    const size_t* input_shape = vollo_rt_model_input_shape(ctx, model_index, 0);

    while (*input_shape != 0) {
      fprintf(stderr, "%ld", *input_shape);
      num_input_elems *= *input_shape;
      input_shape++;

      if (*input_shape != 0) {
        fprintf(stderr, ", ");
      } else {
        fprintf(stderr, "]\n");
      }
    }
  }

  fprintf(stderr, "  %ld output with shape: [", model_num_outputs);

  // Initialised to 1 to get the product of the shape dims
  size_t num_output_elems = 1;
  {
    const size_t* output_shape = vollo_rt_model_output_shape(ctx, model_index, 0);
    while (*output_shape != 0) {
      fprintf(stderr, "%ld", *output_shape);
      num_output_elems *= *output_shape;
      output_shape++;

      if (*output_shape != 0) {
        fprintf(stderr, ", ");
      } else {
        fprintf(stderr, "]\n");
      }
    }
  }

  if (vollo_rt_model_input_streaming_dim(ctx, model_index, 0) >= 0) {
    fprintf(stderr, "  The model is streaming\n");
  }

  assert(vollo_rt_model_input_num_elements(ctx, model_index, 0) == num_input_elems);
  assert(vollo_rt_model_output_num_elements(ctx, model_index, 0) == num_output_elems);

  if (options.input_path != NULL) {
    // Check that the input has the number of input elements that the model expects
    assert(input_array.buffer_len == num_input_elems);
    const size_t* input_shape = vollo_rt_model_input_shape(ctx, model_index, 0);

    // Check that the input has the shape of input that the model expects
    for (size_t i = 0; i < input_array.shape_len; i++) {
      assert(input_array.shape[i] == *input_shape);
      input_shape++;
    }
  }

  //////////////////////////////////////////////////
  // Setup inputs/outputs buffers

  // Number of input vectors
  // When random input is used, we randomly select a vector of random data for each inference
  size_t num_inputs = options.random_input ? NUM_RANDOM_INPUT_VECTORS : 1;
  bf16** inputs = (bf16**)malloc(sizeof(bf16*) * num_inputs);
  float** inputs_fp32 = (float**)malloc(sizeof(float*) * num_inputs);

  for (size_t i = 0; i < num_inputs; i++) {
    inputs[i] = (bf16*)malloc(sizeof(bf16) * num_input_elems);
    inputs_fp32[i] = (float*)malloc(sizeof(float) * num_input_elems);

    for (size_t j = 0; j < num_input_elems; j++) {
      if (options.input_path != NULL) {
        inputs[i][j] = float_to_bf16(input_array.buffer[j]);
        inputs_fp32[i][j] = input_array.buffer[j];
      } else {
        inputs[i][j] = options.random_input ? rand_bf16() : 0x3f80;  // 1.0 as a bf16
        inputs_fp32[i][j] = options.random_input ? rand_float() : 1.0f;
      }
    }
  }

  bf16* output = (bf16*)malloc(sizeof(bf16) * num_output_elems);
  float* output_fp32 = (float*)malloc(sizeof(float) * num_output_elems);

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
        const float* input_arr[1] = {inputs_fp32[input_ix]};
        float* output_arr[1] = {output_fp32};

        EXIT_ON_ERROR(vollo_rt_add_job_fp32(ctx, model_index, user_ctx, input_arr, output_arr));
      } else {
        const bf16* input_arr[1] = {inputs[input_ix]};
        bf16* output_arr[1] = {output};

        EXIT_ON_ERROR(vollo_rt_add_job_bf16(ctx, model_index, user_ctx, input_arr, output_arr));
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
        // if it is not warmup
        if (inf_completed >= options.num_warmup_inferences) {
          size_t ix = inf_completed - options.num_warmup_inferences;

          latencies[ix] = diff_timespec_ns(start_times[ix], job_completed_time);
        }

        inf_completed++;
      }
    }
  }

  start_compute_time = start_times[0];
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  //////////////////////////////////////////////////
  // Summarize latencies

  qsort(latencies, options.num_inferences, sizeof(double), compare_double);

  double sum_latencies_ns = 0.0;
  for (int i = 0; i < (int)options.num_inferences; i++) {
    sum_latencies_ns += latencies[i];
  }

  double mean_latency_ns = sum_latencies_ns / (double)options.num_inferences;
  double median_latency_ns = latencies[options.num_inferences / 2];
  double p99_latency_ns = latencies[(99 * options.num_inferences) / 100];
  double worst_latency_ns = latencies[options.num_inferences - 1];

  double setup_time = diff_timespec_ns(start_setup_time, start_warmup_time) / NANOSECONDS;
  double warmup_time = diff_timespec_ns(start_warmup_time, start_compute_time) / NANOSECONDS;
  double compute_time = diff_timespec_ns(start_compute_time, end_time) / NANOSECONDS;
  double throughput = (double)options.num_inferences / compute_time;

  fprintf(stderr, "Done\n");

  if (options.json) {
    printf("{\n");
    printf("  \"options\": {\n");
    printf("    \"max_concurrent_jobs\": %ld,\n", options.max_concurrent_jobs);
    printf("    \"num_inferences\": %ld\n", options.num_inferences);
    printf("  },\n");
    printf("  \"metrics\": {\n");
    printf("    \"time\": {\n");
    printf("      \"setup\": %f,\n", setup_time);
    printf("      \"warmup\": %f,\n", warmup_time);
    printf("      \"compute\": %f\n", compute_time);
    printf("    },\n");
    printf("    \"throughput\": %f,\n", throughput);
    printf("    \"latency_us\": {\n");
    printf("      \"mean\": %f,\n", mean_latency_ns / 1000);
    printf("      \"median\": %f,\n", median_latency_ns / 1000);
    printf("      \"p99\": %f,\n", p99_latency_ns / 1000);
    printf("      \"worst\": %f\n", worst_latency_ns / 1000);
    printf("    }\n");
    printf("  }\n");
    printf("}\n");
  } else {
    printf("Ran %ld inferences in %f s with:\n", options.num_inferences, compute_time);
    printf("  mean latency of %f us\n", mean_latency_ns / 1000);
    printf("  99%% latency of %f us\n", p99_latency_ns / 1000);
    printf("  throughput of %f inf/s\n", throughput);
  }

  //////////////////////////////////////////////////
  // Serialize output

  if (options.output_path != NULL) {
    NpyArray output_array;
    output_array.buffer = (float*)malloc(sizeof(float) * num_output_elems);
    output_array.buffer_len = num_output_elems;
    {
      const size_t* output_shape = vollo_rt_model_output_shape(ctx, model_index, 0);

      output_array.shape_len = 0;
      while (*output_shape != 0) {
        output_array.shape[output_array.shape_len] = *output_shape;
        output_array.shape_len++;
        output_shape++;
      }
    }

    for (size_t i = 0; i < num_output_elems; i++) {
      if (options.fp32_api) {
        output_array.buffer[i] = output_fp32[i];
      } else {
        output_array.buffer[i] = bf16_to_float(output[i]);
      }
    }

    write_npy(options.output_path, output_array);

    free_npy(output_array);
  }

  //////////////////////////////////////////////////
  // Teardown/Cleanup

  free(latencies);
  free(start_times);
  free(output);
  free(output_fp32);
  for (size_t i = 0; i < num_inputs; i++) {
    free(inputs[i]);
    free(inputs_fp32[i]);
  }
  free(inputs);
  free(inputs_fp32);

  vollo_rt_destroy(ctx);

  if (options.input_path != NULL) {
    free_npy(input_array);
  }
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
    "    -i, --num-inferences\n"
    "        Number of inferences to compute\n"
    "        Defaults to 10_000\n"
    "\n"

    "    -F, --fp32-api\n"
    "        Use the fp32 version of the API (compute is still done in bf16)\n"
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

    "    -j, --json\n"
    "        Output detailed measurements in JSON\n"
    "\n"

    "    -f, --input\n"
    "        Path to an input file (.npy format with bfloat16 as dtype)\n"
    "\n"

    "    -o, --output\n"
    "        Path to an output file (.npy format with bfloat16 as dtype)\n"
    "        Note: Only the first output is serialized to a file even when\n"
    "        num_inferences > 1\n"
    "\n"

    "    -h, --help\n"
    "        Prints this help information\n"
    "\n",
    example_program);
}

int main(int argc, char** argv) {
  ExampleOptions options;
  options.program_path = "";
  options.fp32_api = false;
  options.random_input = false;
  options.max_concurrent_jobs = 1;
  options.num_inferences = 10000;
  options.num_warmup_inferences = 10000;
  options.json = false;
  options.input_path = NULL;
  options.output_path = NULL;

  // Seed the random number generator
  srand((uint32_t)time(NULL));

  // Parse example options
  static struct option long_options[] = {
    {"num-inferences", required_argument, 0, 'i'},
    {"fp32-api", no_argument, 0, 'F'},
    {"random", no_argument, 0, 'r'},
    {"max-concurrent-jobs", required_argument, 0, 'c'},
    {"num-warmup-inferences", required_argument, 0, 'w'},
    {"json", no_argument, 0, 'j'},
    {"input", required_argument, 0, 'f'},
    {"output", required_argument, 0, 'o'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0},
  };

  int opt = 0;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, "i:rc:w:jf:o:h", long_options, &long_index)) != -1) {
    switch (opt) {
    case 'i': options.num_inferences = (size_t)strtoul(optarg, NULL, 10); break;
    case 'F': options.fp32_api = true; break;
    case 'r': options.random_input = true; break;
    case 'c': options.max_concurrent_jobs = (size_t)strtoul(optarg, NULL, 10); break;
    case 'w': options.num_warmup_inferences = (size_t)strtoul(optarg, NULL, 10); break;
    case 'j': options.json = true; break;
    case 'f': options.input_path = optarg; break;
    case 'o': options.output_path = optarg; break;
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

  if (options.random_input && options.input_path != NULL) {
    fprintf(stderr, "Options -r,--random and -f,--input are not compatible\n");
    exit(EXIT_FAILURE);
  }

  vollo_example(options);
}

// Generate a random float in the range ± 1.0
static float rand_float() {
  return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

// Generate a random bf16 in the range ± 1.0
static bf16 rand_bf16() {
  float x = rand_float();
  uint32_t x_int;
  memcpy(&x_int, &x, sizeof(uint32_t));
  return (bf16)(x_int >> 16);
}

// Convert from bf16 to float
static float bf16_to_float(bf16 x) {
  uint32_t y = ((uint32_t)x) << 16;
  float y_float;
  memcpy(&y_float, &y, sizeof(float));
  return y_float;
}

// Convert from float to bf16
// This conversion truncates the mantissa instead of rounding
static bf16 float_to_bf16(float x) {
  uint32_t x_int;
  memcpy(&x_int, &x, sizeof(float));
  return (bf16)(x_int >> 16);
}

// Compare two doubles
static int compare_double(const void* a, const void* b) {
  if (*(double*)a > *(double*)b)
    return 1;
  else if (*(double*)a < *(double*)b)
    return -1;
  else
    return 0;
}

static double diff_timespec_ns(struct timespec from, struct timespec to) {
  return (double)((long long)(to.tv_sec - from.tv_sec) * NANOSECONDS + (to.tv_nsec - from.tv_nsec));
}
