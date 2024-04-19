// Copyright(C) 2024 Myrtle Software Ltd. All rights reserved.

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Functions in vollo-rt that can return an error return `vollo_rt_error_t`.
 * NULL is returned where there are no errors, otherwise it is a null-terminated string containing
 * an error message.
 *
 * Error messages are owned by vollo-rt and can be freed with `vollo_rt_destroy_err`
 */
typedef const char* vollo_rt_error_t;

/**
 * A context for performing computation on a vollo. The context is constructed with `vollo_rt_init`
 * and destroyed with `vollo_rt_destroy`.
 */
typedef void* vollo_rt_context_t;

/**
 * bfloat16: Brain Floating Point Format
 * https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
 *
 * Note: Make sure to NOT use C literals to create values of this type,
 * instead convert from float (by truncating or rounding as appropriate)
 */
typedef uint16_t bf16;

/**
 * Return a static string of the vollo-rt version.
 */
const char* vollo_rt_version(void);

/**
 * All APIs return the error as a c string. To prevent leaking the memory, destroy it afterwards.
 */
void vollo_rt_destroy_err(vollo_rt_error_t err);

/**
 * Initialise the vollo-rt context. This must be called before any other vollo-rt functions.
 *
 * Logging level can be configured by setting the environment variable `VOLLO_RT_LOG` to one of:
 * "error", "warn", "info", "debug", or "trace"
 */
vollo_rt_error_t vollo_rt_init(vollo_rt_context_t* context_ptr);

/**
 * Destroy vollo-rt context, releasing its associated resources.
 */
void vollo_rt_destroy(vollo_rt_context_t vollo);

/**
 * Add an accelerator.
 * The accelerator is specified by its index. The index refers to an accelerator in the sorted list
 * of PCI addresses. This should be called after `vollo_rt_init` but before `vollo_rt_load_program`
 */
vollo_rt_error_t vollo_rt_add_accelerator(vollo_rt_context_t vollo, size_t accelerator_index);

/**
 * Get the number of cores of a Vollo accelerator
 *
 * Requirements (panics otherwise):
 * - The accelerator at `accelerator_index` has already been added to context
 *   with `vollo_rt_add_accelerator`
 */
size_t vollo_rt_accelerator_num_cores(vollo_rt_context_t vollo, size_t accelerator_index);

/**
 * Get the block size of a Vollo accelerator
 *
 * Requirements (panics otherwise):
 * - The accelerator at `accelerator_index` has already been added to context
 *   with `vollo_rt_add_accelerator`
 */
size_t vollo_rt_accelerator_block_size(vollo_rt_context_t vollo, size_t accelerator_index);

/**
 * Load a program onto the Vollo accelerators.
 * This should be called after `vollo_rt_add_accelerator`
 *
 * A Vollo program is generated by the Vollo compiler, it is typically named
 * "<program_name>.vollo".
 * The program is intended for a specific hw_config (number of accelerators,
 * cores and other HW configuration options), this function will return an
 * error if any accelerator configuration is incompatible with the program.
 * Once loaded, the program provides inference for several models concurrently.
 *
 * Note: This should only be called once per `vollo_rt_context_t`, as such if
 * a program needs to be changed or reset, first `vollo_rt_destroy` the current
 * context, then start a new context with `vollo_rt_init`.
 */
vollo_rt_error_t vollo_rt_load_program(vollo_rt_context_t vollo, const char* program_path);

/**
 * Inspect the number of models in the program loaded onto the vollo.
 *
 * Programs can contain multiple models, a `model_index` is used to select a
 * specific model
 */
size_t vollo_rt_num_models(vollo_rt_context_t vollo);

/**
 * Get the name of a model (or NULL if no name was set)
 *
 * The returned string is owned by the `vollo_rt_context_t` and lives for as
 * long as the loaded program.
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 */
const char* vollo_rt_model_name(vollo_rt_context_t vollo, size_t model_index);

/**
 * Get the number of inputs of a model
 *
 * Each input has its own distinct shape
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 */
size_t vollo_rt_model_num_inputs(vollo_rt_context_t vollo, size_t model_index);

/**
 * Get the number of outputs of a model
 *
 * Each output has its own distinct shape
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 */
size_t vollo_rt_model_num_outputs(vollo_rt_context_t vollo, size_t model_index);

/**
 * Get the shape for input at a given index
 *
 * The return value is a 0 terminated array of dims containing the input shape
 * The value lives for as long as the model
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 * - `input_index < vollo_rt_model_num_inputs`
 */
const size_t* vollo_rt_model_input_shape(
  vollo_rt_context_t vollo, size_t model_index, size_t input_index);

/**
 * Get the shape for output at a given index
 *
 * The return value is a 0 terminated array of dims containing the output shape
 * The value lives for as long as the model
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 * - `output_index < vollo_rt_model_num_outputs`
 */
const size_t* vollo_rt_model_output_shape(
  vollo_rt_context_t vollo, size_t model_index, size_t output_index);

/**
 * Get the number of elements for input at a given index
 *
 * This is simply the product of the dimensions returned by `vollo_rt_model_input_shape`,
 * it is provided to make it easier to allocate the correct number of elements.
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 * - `input_index < vollo_rt_model_num_inputs`
 */
size_t vollo_rt_model_input_num_elements(
  vollo_rt_context_t vollo, size_t model_index, size_t input_index);

/**
 * Get the number of elements for output at a given index
 *
 * This is simply the product of the dimensions returned by `vollo_rt_model_output_shape`,
 * it is provided to make it easier to allocate the correct number of elements.
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 * - `output_index < vollo_rt_model_num_outputs`
 */
size_t vollo_rt_model_output_num_elements(
  vollo_rt_context_t vollo, size_t model_index, size_t output_index);

/**
 * In a streaming model, the streaming dimension is not part of the shape.
 *
 * - It returns -1 when there is no streaming dimension
 * - It otherwise returns the dim index
 *   For example, for a shape `(a, b, c)` and streaming dim index 1, the full shape is:
 *   `(a, streaming_dim, b, c)`
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 * - `input_index < vollo_rt_model_num_inputs`
 */
int vollo_rt_model_input_streaming_dim(
  vollo_rt_context_t vollo, size_t model_index, size_t input_index);

/**
 * In a streaming model, the streaming dimension is not part of the shape.
 *
 * - It returns -1 when there is no streaming dimension
 * - It otherwise returns the dim index
 *   For example, for a shape `(a, b, c)` and streaming dim index 1, the full shape is:
 *   `(a, streaming_dim, b, c)`
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 * - `output_index < vollo_rt_model_num_outputs`
 */
int vollo_rt_model_output_streaming_dim(
  vollo_rt_context_t vollo, size_t model_index, size_t output_index);

/**
 * Sets up a computation on the vollo accelerator where the inputs and outputs are in brain-float 16
 * format.
 *
 * Note: The computation is only started on the next call to vollo_rt_poll. This way it is possible
 * to set up several computations that are kicked off at the same time.
 *
 * - vollo:
 *     the context that the computation should be run on
 * - model_index:
 *     the model to run
 * - user_ctx:
 *     a user context that will be returned on completion. This can be used to disambiguate when
 *     multiple models are running concurrently.
 *     NOTE: the jobs for a single model are guaranteed to come back in order, but the jobs for
 *     different models are not.
 * - input_data:
 *     a pointer to the start of an array with pointers to the start of the data to each input the
 *     number of inputs is given by `vollo_rt_model_num_inputs` each input length is the product of
 *     the shape given by `vollo_rt_model_input_shape`
 *     (or more convenient: `vollo_rt_model_input_num_elements`)
 *     lifetime:
 *       - The outer array only needs to live until `vollo_rt_add_job_bf16` returns
 *       - The input buffers need to live until `vollo_rt_poll` returns with the completion for
 *         this job
 * - output_data:
 *     a pointer to the start of an array with pointers to the start of the data to each output
 *     buffer the number of outputs is given by `vollo_rt_model_num_outputs` each output length is
 *     the product of the shape given by `vollo_rt_model_output_shape`
 *     (or more convenient: `vollo_rt_model_output_num_elements`)
 *     lifetime:
 *       - The outer array only needs to live until `vollo_rt_add_job_bf16` returns
 *       - The output buffers need to live until `vollo_rt_poll` returns with the completion for
 *         this job
 */
vollo_rt_error_t vollo_rt_add_job_bf16(
  vollo_rt_context_t vollo,
  size_t model_index,
  uint64_t user_ctx,
  const bf16* const* input_data,
  bf16* const* output_data);

/**
 * Sets up a computation on the vollo accelerator where the inputs and outputs are in fp32 format.
 *
 * Note:
 * - The computation will still be performed in bf16 but the driver will perform the conversion.
 * - The computation is only started on the next call to vollo_rt_poll. This way it is possible
 *   to set up several computations that are kicked off at the same time.
 *
 * - vollo:
 *     the context that the computation should be run on
 * - model_index:
 *     the model to run
 * - user_ctx:
 *     a user context that will be returned on completion. This can be used to disambiguate when
 *     multiple models are running concurrently.
 *     NOTE: the jobs for a single model are guaranteed to come back in order, but the jobs for
 *     different models are not.
 * - input_data:
 *     a pointer to the start of an array with pointers to the start of the data to each input the
 *     number of inputs is given by `vollo_rt_model_num_inputs` each input length is the product of
 *     the shape given by `vollo_rt_model_input_shape`
 *     (or more convenient: `vollo_rt_model_input_num_elements`)
 *     lifetime:
 *       - The outer array only needs to live until `vollo_rt_add_job_fp32` returns
 *       - The input buffers need to live until `vollo_rt_poll` returns with the completion for
 *         this job
 * - output_data:
 *     a pointer to the start of an array with pointers to the start of the data to each output
 *     buffer the number of outputs is given by `vollo_rt_model_num_outputs` each output length is
 *     the product of the shape given by `vollo_rt_model_output_shape`
 *     (or more convenient: `vollo_rt_model_output_num_elements`)
 *     lifetime:
 *       - The outer array only needs to live until `vollo_rt_add_job_fp32` returns
 *       - The output buffers need to live until `vollo_rt_poll` returns with the completion for
 *         this job
 */
vollo_rt_error_t vollo_rt_add_job_fp32(
  vollo_rt_context_t vollo,
  size_t model_index,
  uint64_t user_ctx,
  const float* const* input_data,
  float* const* output_data);

/**
 * Poll the vollo accelerator for completion.
 *
 * Note: Polling also initiates transfers for new jobs, so poll must be called
 * before any progress on these new jobs can be made.
 *
 *   num_completed: out: the number of completed user_ctx returned
 *   returned_user_ctx: buffer for the returned user_ctx of completed jobs, this will only be
 *                      valid until the next call to vollo_rt_poll.
 *
 * Larger IO might be split over multiple calls to `vollo_rt_poll`. Since the optimal chunking
 * size depends on the program and the system Vollo is running on, the maximum chunk size is
 * configurable when loading a program with environment variables `VOLLO_IN_MAX_CHUNK_SIZE`
 * and `VOLLO_OUT_MAX_CHUNK_SIZE` (both default to 8192 values)
 */
vollo_rt_error_t vollo_rt_poll(
  vollo_rt_context_t vollo, size_t* num_completed, const uint64_t** returned_user_ctx);
