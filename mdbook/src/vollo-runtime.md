# VOLLO Runtime

The VOLLO runtime provides a low latency asynchronous inference API for timing
critical inference requests.

An example C program that uses the VOLLO runtime API has been included in the
installation in the `example/` directory.

## C API

### Initialisation

A vollo context is created by calling `vollo_init`.
Add an accelerator by using the `vollo_add_accelerator` function.

```c
  /**
   * Initialise the vollo context. This must be called before any other vollo functions.
   */
  const char* vollo_init(vollo_context_t* vollo_ptr);

  /**
   * Add an accelerator.
   * The accelerator is specified by its index. The index refers to an accelerator in the sorted list
   * of PCI addresses. This should be called after `vollo_init` but before `vollo_load_program` or
   * `vollo_load_config`.
   */
  const char* vollo_add_accelerator(vollo_context_t vollo, uintptr_t accelerator_index);

  /**
   * Destroy the vollo.
   */
  void vollo_destroy(vollo_context_t vollo);
```

### Loading a program

A program is loaded onto the vollo using the `vollo_load_program` function.

```c
  /**
   * Load a pre-compiled program onto the board.
   * This should be called after `vollo_add_accelerator`, and must be called before
   * `vollo_init_model`.
   */
  const char* vollo_load_program(vollo_context_t vollo, const char* program_path);

  /**
   * Initialise a model. The should be called after `vollo_load_program` and
   * must be called before `vollo_add_job_bf16` or `vollo_add_job_fp32`.
   */
  const char* vollo_init_model(
    vollo_context_t vollo, uintptr_t model_index, vollo_model_t* model_handle);
```

### Running inference

The loaded model creates an inference stream handle.
The interface offers an asynchronous result function so that input inference
requests can be made as fast as the system can support, prior to the return of
output data.

```c
  /**
   * Run a computation on the vollo where the inputs and outputs are in brain-float 16 format.
   *
   *   vollo: the context that the computation should be run on
   *   model: the program to run
   *   user_ctx: a user context that will be returned on completion
   *   num_timesteps: the number of timesteps in the input
   *   input_data: row major 2D array input data where each row is a timestep
   *   outputs: a buffer where a single row of the final output from the final timestep will
   *   be written
   */
  const char* vollo_add_job_bf16(
    vollo_context_t vollo,
    vollo_model_t model,
    uint64_t user_ctx,
    uint32_t num_timesteps,
    uint32_t num_features,
    const bf16* input_data,
    bf16* outputs);

  /**
   * Run a computation on the vollo with fp32 inputs and outputs. NOTE: the computation
   * will still be performed in bf16 but the driver will perform the conversion.
   *
   *   vollo: the context that the computation should be run on
   *   model: the model to run
   *   user_ctx: a user context that will be returned on completion
   *   num_timesteps: the number of timesteps in the input
   *   input_data: row major 2D array input data where each row is a timestep
   *   outputs: a buffer where a single row of the final output from the final timestep will
   *   be written
   */
  const char* vollo_add_job_fp32(
    vollo_context_t vollo,
    vollo_model_t model,
    uint64_t user_ctx,
    uint32_t num_timesteps,
    uint32_t num_features,
    const float* input_data,
    float* outputs);

  /**
   * Poll the vollo for completion. Note that polling also initiates transfers so you must
   * poll before anything happens.
   *
   * num_completed: out: the number of completed user_ctx returned
   * returned_user_ctx: buffer for the returned user_ctx of completed jobs, this will only be
   * valid until the next call to vollo_poll.
   */
  const char* vollo_poll(
    vollo_context_t vollo, uintptr_t* num_completed, const uint64_t** returned_user_ctx);
```
