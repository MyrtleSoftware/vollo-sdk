#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */

#include "utils.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <vollo-rt.h>

// A small wrapper around the asynchronous Vollo RT API to block on a single inference
// This assume a single model with a single input and output tensor
static void single_shot_inference(vollo_rt_context_t ctx, const float* input, float* output) {
  size_t model_index = 0;

  const float* inputs[1] = {input};
  float* outputs[1] = {output};

  // user_ctx is not needed when doing single shot inferences
  // it can be used when doing multiple jobs concurrently to keep track of which jobs completed
  uint64_t user_ctx = 0;

  // Register a new job
  EXIT_ON_ERROR(vollo_rt_add_job_fp32(ctx, model_index, user_ctx, inputs, outputs));

  // Poll until completion
  size_t num_completed = 0;
  const uint64_t* completed_buffer = NULL;
  size_t poll_count = 0;

  while (num_completed == 0) {
    EXIT_ON_ERROR(vollo_rt_poll(ctx, &num_completed, &completed_buffer));

    poll_count++;
    if (poll_count > 100000000) {
      fprintf(stderr, "timed out while polling\n");
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char** argv) {
  const char* device_spec = default_device_spec();

  static struct option long_options[] = {
    {"device", required_argument, 0, 'd'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0},
  };

  int opt = 0;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, "d:h", long_options, &long_index)) != -1) {
    switch (opt) {
    case 'd': device_spec = optarg; break;
    case 'h':
      printf("USAGE:\n    %s [--device <SPEC>]\n", argv[0]);
      printf("Defaults to $VOLLO_CARD_BDF if set, otherwise --device 0\n");
      return EXIT_SUCCESS;
    default: return EXIT_FAILURE;
    }
  }

  //////////////////////////////////////////////////
  // Init
  vollo_rt_context_t ctx;
  EXIT_ON_ERROR(vollo_rt_init(&ctx));

  //////////////////////////////////////////////////
  // Add accelerators
  EXIT_ON_ERROR(vollo_rt_add_device(ctx, 0, device_spec));

  //////////////////////////////////////////////////
  // Load program
  {
    char program_path[64];
    snprintf(
      program_path,
      sizeof(program_path),
      "./identity_b%zu.vollo",
      vollo_rt_accelerator_block_size(ctx, 0));
    EXIT_ON_ERROR(vollo_rt_load_program(ctx, program_path));
  }

  //////////////////////////////////////////////////
  // Setup inputs and outputs

  size_t model_index = 0;

  // Assert model only has a single input and a single output tensor
  ALWAYS_ASSERT(vollo_rt_model_num_inputs(ctx, model_index) == 1);
  ALWAYS_ASSERT(vollo_rt_model_num_outputs(ctx, model_index) == 1);

  ALWAYS_ASSERT(vollo_rt_model_input_num_elements(ctx, model_index, 0) == 128);
  ALWAYS_ASSERT(vollo_rt_model_output_num_elements(ctx, model_index, 0) == 128);

  float input_tensor[128];
  float output_tensor[128];

  for (size_t i = 0; i < 128; i++) {
    input_tensor[i] = 42.0;
  }

  //////////////////////////////////////////////////
  // Run an inference

  single_shot_inference(ctx, input_tensor, output_tensor);

  //////////////////////////////////////////////////
  // Print outputs

  printf("Output values: [");
  for (size_t i = 0; i < 128; i++) {
    if (i % 8 == 0) {
      printf("\n  ");
    }

    printf("%.1f, ", output_tensor[i]);
  }
  printf("\n]\n");

  //////////////////////////////////////////////////
  // Release resources / Cleanup
  vollo_rt_destroy(ctx);

  return 0;
}
