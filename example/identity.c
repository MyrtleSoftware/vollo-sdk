#include <assert.h>
#include <stdio.h>
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
    if (poll_count > 1000000) {
      EXIT_ON_ERROR("Timed out while polling");
    }
  }
}

int main(void) {
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

  {
    printf("Trying to load the block_size 32 identity Vollo program\n");
    vollo_rt_error_t err = vollo_rt_load_program(ctx, "./identity_b32.vollo");

    if (err != NULL) {
      vollo_rt_destroy_err(err);

      printf("It didn't work, trying to load the block_size 64 identity Vollo program instead\n");
      EXIT_ON_ERROR(vollo_rt_load_program(ctx, "./identity_b64.vollo"));
    }

    printf("Success");
  }

  //////////////////////////////////////////////////
  // Setup inputs and outputs

  size_t model_index = 0;

  // Assert model only has a single input and a single output tensor
  assert(vollo_rt_model_num_inputs(ctx, model_index) == 1);
  assert(vollo_rt_model_num_outputs(ctx, model_index) == 1);

  assert(vollo_rt_model_input_num_elements(ctx, model_index, 0) == 128);
  assert(vollo_rt_model_output_num_elements(ctx, model_index, 0) == 128);

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
