#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */
#include "npy.h"
#include "utils.h"

#include <assert.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vollo-rt.h>

static void full_input_inference_start(vollo_rt_context_t, const bf16*, bf16**);
static void partial_input_inference_start(
  vollo_rt_context_t, uint32_t, const uint32_t*, const bf16*, bf16**);
static void block_until_completion(vollo_rt_context_t);

void print_help(const char* prog) {
  printf(
    "\n"
    "USAGE:\n    %s [OPTIONS] <VOLLO_PROGRAM>\n\n"
    "ARGS:\n"
    "    <VOLLO_PROGRAM>\n"
    "        Path to the Vollo program file (e.g: ./program.vollo)\n"
    "\n"

    "OPTIONS:\n"

    "    -o, --output-dir\n"
    "        Path to an output directory\n"
    "        Each inference for each model output is saved in a separate file in .npy format with\n"
    "        float32 as dtype\n"
    "\n"

    "    -i, --num-inferences\n"
    "        Number of inferences to compute\n"
    "        Defaults to 1_000\n"
    "\n"

    "    -t, --threshold-partial\n"
    "        Number of updates to the input from which to uses the full input API instead of\n"
    "        the partial input API\n"
    "        Defaults to -1 (no limit, always use partial input API)\n"
    "\n"

    "    -h, --help\n"
    "        Prints this help information\n"
    "\n",
    prog);
}

int main(int argc, char** argv) {
  //////////////////////////////////////////////////
  // Parse options from CLI
  char* program_path = NULL;
  char* output_dir = NULL;
  size_t num_inferences = 1000;
  long int threshold_partial_updates = -1;

  static struct option long_options[] = {
    {"output-dir", required_argument, 0, 'o'},
    {"num-inferences", required_argument, 0, 'i'},
    {"threshold-partial", required_argument, 0, 't'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0},
  };

  int opt = 0;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, "o:i:t:h", long_options, &long_index)) != -1) {
    switch (opt) {
    case 'o': output_dir = optarg; break;
    case 'i': num_inferences = (size_t)strtoul(optarg, NULL, 10); break;
    case 't': threshold_partial_updates = strtol(optarg, NULL, 10); break;
    default: print_help(argv[0]); exit(opt == 'h' ? EXIT_SUCCESS : EXIT_FAILURE);
    }
  }

  if (optind == (argc - 1)) {
    program_path = argv[optind];
    fprintf(stderr, "Using program: \"%s\"\n", program_path);
    fprintf(stderr, "  num-inferences: %ld\n", num_inferences);
    fprintf(stderr, "  threshold-partial: %ld\n", threshold_partial_updates);
    if (output_dir != NULL) {
      fprintf(stderr, "  output dir: \"%s\"\n", output_dir);
    }
  } else {
    print_help(argv[0]);
    exit(EXIT_FAILURE);
  }

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
  EXIT_ON_ERROR(vollo_rt_load_program(ctx, program_path));

  //////////////////////////////////////////////////
  // Setup inputs and outputs

  // Assert it is a single model program (partial update not supported otherwise)
  assert(vollo_rt_num_models(ctx) == 1);
  size_t model_index = 0;

  // Assert model only has a single input
  assert(vollo_rt_model_num_inputs(ctx, model_index) == 1);

  size_t num_input_elements = vollo_rt_model_input_num_elements(ctx, model_index, 0);

  size_t model_num_outputs = vollo_rt_model_num_outputs(ctx, model_index);

  bf16* input_tensor = vollo_rt_get_raw_buffer(ctx, num_input_elements);
  bf16** output_tensors = (bf16**)malloc(model_num_outputs * sizeof(bf16*));
  for (size_t i = 0; i < model_num_outputs; i++) {
    size_t num_output_elements = vollo_rt_model_output_num_elements(ctx, model_index, i);
    output_tensors[i] = vollo_rt_get_raw_buffer(ctx, num_output_elements);
  }

  bf16*** outputs = NULL;
  if (output_dir != NULL) {
    outputs = (bf16***)malloc(num_inferences * sizeof(bf16**));

    for (size_t i = 0; i < num_inferences; i++) {
      outputs[i] = (bf16**)malloc(model_num_outputs * sizeof(bf16*));

      for (size_t j = 0; j < model_num_outputs; j++) {
        size_t num_output_elements = vollo_rt_model_output_num_elements(ctx, model_index, j);
        outputs[i][j] = (bf16*)malloc(num_output_elements * sizeof(bf16));
      }
    }
  }

  // Allocating num_input_elements is excessive for partial updates
  // (when updating all inputs it is more efficient to use the full input API)
  //
  // However we do this for the example to simplify the random index selection (with no duplicates)
  uint32_t num_partial_updates = 0;
  uint32_t* partial_update_indices = (uint32_t*)malloc(num_input_elements * sizeof(uint32_t));
  bf16* partial_update_values = (bf16*)malloc(num_input_elements * sizeof(bf16));

  double* latencies = (double*)malloc(sizeof(double) * num_inferences);
  struct timespec start_time, completed_time;

  // Seed the randomness to be able to compare runs
  srand(0);

  for (size_t i = 0; i < num_input_elements; i++) {
    input_tensor[i] = rand_bf16();
    partial_update_indices[i] = (uint32_t)i;  // initialise update indices to ensure no duplicates
  }

  //////////////////////////////////////////////////
  // Run inferences
  for (size_t i = 0; i < num_inferences; i++) {
    if (output_dir != NULL) {
      // If we're saving the outputs, update output_tensors to the next buffers
      for (size_t j = 0; j < model_num_outputs; j++) {
        output_tensors[j] = outputs[i][j];
      }
    }

    //////////////////////////////////////////////////
    // Random update to the input for the next inference

    uint32_t extra_magnitude = (uint32_t)rand();
    uint32_t extra = 1 << 4;
    while ((extra_magnitude & extra) == 0 && extra != 0) {
      extra <<= 1;
    }
    num_partial_updates = (uint32_t)rand() & (extra - 1);

    if (num_partial_updates > num_input_elements) {
      num_partial_updates = (uint32_t)num_input_elements;
    }

    partial_rand_shuffle(num_partial_updates, num_input_elements, partial_update_indices);
    for (size_t j = 0; j < num_partial_updates; j++) {
      partial_update_values[j] = rand_bf16();

      // Apply random update to input tensor
      input_tensor[partial_update_indices[j]] = partial_update_values[j];
    }

    //////////////////////////////////////////////////
    // Run an inference

    // Do a full input inference for the first inference or for any inference where the number of
    // updates exceeds the threshold
    bool do_full_input_inference
      = i == 0
        || (threshold_partial_updates >= 0 && threshold_partial_updates < num_partial_updates);

    // Start an inference
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    if (do_full_input_inference) {
      // Apply the input updates to input tensor
      for (size_t j = 0; j < num_partial_updates; j++) {
        input_tensor[partial_update_indices[j]] = partial_update_values[j];
      }

      full_input_inference_start(ctx, input_tensor, output_tensors);

    } else {
      partial_input_inference_start(
        ctx, num_partial_updates, partial_update_indices, partial_update_values, output_tensors);

      // Apply the input updates to input tensor for future inferences with full inputs while we
      // compute the current inference with partially update input
      for (size_t j = 0; j < num_partial_updates; j++) {
        input_tensor[partial_update_indices[j]] = partial_update_values[j];
      }
    }

    // Wait for completion of the inference
    block_until_completion(ctx);

    clock_gettime(CLOCK_MONOTONIC, &completed_time);
    latencies[i] = diff_timespec_ns(start_time, completed_time);
  }

  //////////////////////////////////////////////////
  // Summarize latencies

  latency_summary summary = summarize_latencies(num_inferences, latencies);

  printf("{\n");
  printf("  \"latency_us\": {\n");
  printf("    \"mean\": %f,\n", summary.mean_latency_ns / 1000);
  printf("    \"best\": %f,\n", summary.best_latency_ns / 1000);
  printf("    \"median\": %f,\n", summary.median_latency_ns / 1000);
  printf("    \"p99\": %f,\n", summary.p99_latency_ns / 1000);
  printf("    \"worst\": %f\n", summary.worst_latency_ns / 1000);
  printf("  }\n");
  printf("}\n");

  //////////////////////////////////////////////////
  // Write outputs to files

  if (output_dir) {
    if (mkdir(output_dir, 0700) == -1) {
      perror("Creating output directory");
      exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < model_num_outputs; i++) {
      size_t num_output_elements = vollo_rt_model_output_num_elements(ctx, model_index, i);

      NpyArray output_array = {0};
      output_array.buffer = (float*)malloc(num_output_elements * sizeof(float));
      output_array.buffer_len = num_output_elements;
      {
        const size_t* output_shape = vollo_rt_model_output_shape(ctx, model_index, i);
        const uint8_t output_shape_len
          = (uint8_t)vollo_rt_model_output_shape_len(ctx, model_index, i);

        output_array.shape_len = output_shape_len;
        for (uint8_t output_shape_ix = 0; output_shape_ix < output_shape_len; output_shape_ix++) {
          output_array.shape[output_shape_ix] = (size_t)*output_shape;
          output_shape++;
        }
      }

      for (size_t j = 0; j < num_inferences; j++) {
        char output_path[1024];
        int ret = snprintf(output_path, 1024, "%s/output_%05ld_%05ld.npy", output_dir, i, j);
        if (ret < 0 || ret >= 1024) {
          fprintf(stderr, "Error creating the output file");
          exit(EXIT_FAILURE);
        }

        for (size_t k = 0; k < num_output_elements; k++) {
          output_array.buffer[k] = bf16_to_float(outputs[j][i][k]);
        }

        write_npy(output_path, output_array);
      }

      free_npy(output_array);
    }
  }

  //////////////////////////////////////////////////
  // Release resources / Cleanup

  if (output_dir) {
    for (size_t i = 0; i < num_inferences; i++) {
      for (size_t j = 0; j < model_num_outputs; j++) {
        free(outputs[i][j]);
      }
      free(outputs[i]);
    }
    free(outputs);
  }
  free(output_tensors);

  free(latencies);
  free(partial_update_indices);
  free(partial_update_values);

  vollo_rt_destroy(ctx);

  return 0;
}

static void full_input_inference_start(vollo_rt_context_t ctx, const bf16* input, bf16** outputs) {
  size_t model_index = 0;

  const bf16* input_data[1] = {input};

  // user_ctx is not needed when doing single shot inferences
  // it can be used when doing multiple jobs concurrently to keep track of which jobs completed
  uint64_t user_ctx = 0;

  // Register a new job
  EXIT_ON_ERROR(vollo_rt_add_job_bf16(ctx, model_index, user_ctx, input_data, outputs));

  // Single poll to start new job
  size_t num_completed = 0;
  const uint64_t* completed_buffer = NULL;
  EXIT_ON_ERROR(vollo_rt_poll(ctx, &num_completed, &completed_buffer));

  // This example doesn't expect to have any concurrently started jobs reaching completion now
  assert(num_completed == 0);
}

static void partial_input_inference_start(
  vollo_rt_context_t ctx,
  uint32_t num_input_updates,
  const uint32_t* input_update_indices,
  const bf16* input_update_values,
  bf16** outputs) {
  size_t model_index = 0;

  // user_ctx is not needed when doing single shot inferences
  // it can be used when doing multiple jobs concurrently to keep track of which jobs completed
  uint64_t user_ctx = 0;

  // Register a new job with partial update input
  EXIT_ON_ERROR(vollo_rt_add_job_bf16_partial_update(
    ctx,
    model_index,
    user_ctx,
    num_input_updates,
    input_update_indices,
    input_update_values,
    outputs));

  // Single poll to start new job
  size_t num_completed = 0;
  const uint64_t* completed_buffer = NULL;
  EXIT_ON_ERROR(vollo_rt_poll(ctx, &num_completed, &completed_buffer));

  // This example doesn't expect to have any concurrently started jobs reaching completion now
  assert(num_completed == 0);
}

#define MAX_POLL_COUNT 1000000

static void block_until_completion(vollo_rt_context_t ctx) {
  // Poll until completion
  size_t num_completed = 0;
  const uint64_t* completed_buffer = NULL;
  size_t poll_count = 0;

  while (num_completed == 0) {
    EXIT_ON_ERROR(vollo_rt_poll(ctx, &num_completed, &completed_buffer));

    poll_count++;
    if (poll_count > MAX_POLL_COUNT) {
      EXIT_ON_ERROR("Timed out while polling");
    }
  }
}
