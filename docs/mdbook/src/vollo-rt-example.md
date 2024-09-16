# Vollo RT Example

The full code for this example can be found in `example/identity.c`.

Here we will work through it step by step.

---

First we need to get hold of a Vollo RT context:

```c
//////////////////////////////////////////////////
// Init
vollo_rt_context_t ctx;
EXIT_ON_ERROR(vollo_rt_init(&ctx));
```

Note: throughout this example we use `EXIT_ON_ERROR`, it is just a convenient way to handle errors

---

Then we need to add accelerators, the `accelerator_index` refers to the index of
the Vollo accelerator in the sorted list of PCI addresses, simply use `0` if you
have a single accelerator, or just want to use the first one.

```c
//////////////////////////////////////////////////
// Add accelerators
size_t accelerator_index = 0;
EXIT_ON_ERROR(vollo_rt_add_accelerator(ctx, accelerator_index));
```

This step will check the accelerator license and make sure the bitstream is the
correct version and compatible with this version of the runtime.

---

Then we load a program:

```c
//////////////////////////////////////////////////
// Load program

// Program for a block_size 64 accelerator
const char* vollo_program_path = "./identity_b64.vollo";
EXIT_ON_ERROR(vollo_rt_load_program(ctx, vollo_program_path));
```

Here we're using a relative path (in the `example` directory) to one of the
example Vollo program, a program that computes the identity function for a tensor of size 128. The program
is specifically for a block_size 64 version of the accelerator such as the
default configuration for the `IA840F` FPGA.

---

Then we setup some inputs and outputs for a single inference:

```c
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
```

We check that the program metadata matches our expectations and we setup an input and output buffer.

---

Then we run a single inference:

```c
//////////////////////////////////////////////////
// Run an inference

single_shot_inference(ctx, input_tensor, output_tensor);
```

Where we define a convenience function to run this type of simple synchronous
inference on top of the asynchronous Vollo RT API:

```c
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
```

This function does 2 things. First it registers a new job with the Vollo RT
context and then it polls in a loop until that job is complete.

For a more thorough overview of how to use this asynchronous API to run multiple
jobs concurrently take a look at `example/example.c`

---

And finally we print out the newly obtained results and cleanup the Vollo RT context:

```c
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
```
