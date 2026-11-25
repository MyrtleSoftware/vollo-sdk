# C API

Make sure the `VOLLO_SDK` environment variable is set by sourcing setup.sh from the Vollo SDK.

```bash
source <path-to-VOLLO_SDK>/setup.sh
```

The Vollo runtime API is a C API with simple types and functions in order to be
straight forward to use from any language with a C FFI.

- Header file: `$VOLLO_SDK/include/vollo-rt.h`
- Dynamic library: `$VOLLO_SDK/lib/libvollo_rt.so`
- Static library: `$VOLLO_SDK/lib/libvollo_rt.a`

It was built against GLIBC version 2.17.

To compile against Vollo RT with a standard C compiler, you can use the following flags:
`-I $VOLLO_SDK/include -L $VOLLO_SDK/lib -lvollo_rt`

These are the main steps (in order) a program using `vollo_rt` will follow:

1. Initialise the Vollo runtime using `vollo_rt_init`
2. Add Vollo accelerators to the runtime using `vollo_rt_add_accelerator`
   (Note: the current release only supports one accelerator)
3. Load a Vollo program onto the Vollo accelerators with `vollo_rt_load_program`
4. Optionally, inspect the metadata about the models in the program using API calls such as
   `vollo_rt_num_models` and `vollo_rt_model_num_inputs`
5. Queue and run inference jobs by first calling `vollo_rt_add_job_bf16` (or
   `vollo_rt_add_job_fp32`) and then polling in a loop for their completion using `vollo_rt_poll`.
   You can queue several jobs before calling `vollo_rt_poll` or add extra jobs at any point.
6. Finally call `vollo_rt_destroy` to release resources.

The API is designed to explicitly return errors when it can to let the user handle them as they see fit.
The metadata functions will instead error out themselves if any of the documented pre-conditions they rely on aren't met.
Any other crash is considered a bug and we would be very grateful if you could tell us about it.

## Errors

Fallible functions return a `vollo_rt_error_t`: NULL for success, otherwise a
NUL-terminated message. Inspect its category with `vollo_rt_error_code(err)`
instead of parsing the text:

| Code | Meaning |
| --- | --- |
| `vollo_rt_error_success` (0) | The error pointer is NULL. |
| `vollo_rt_error_other` (1) | An error without a more specific category. Retrying is not necessarily safe. |
| `vollo_rt_error_would_block` (2) | From `vollo_rt_add_external_input_job`: the job could not be added yet. Nothing was submitted or kept. |
| `vollo_rt_error_timeout` (3) | From `vollo_rt_poll`: an external-input job has made no progress for a while because some of its input has not arrived, and the card is still running. The job stays pending and keeps its buffers; a later poll returns it once the rest of its input arrives. |

More codes may be added; treat an unrecognized code as an error.

Free errors with `vollo_rt_destroy_err`, not `free`. It accepts NULL, so it can be
called whether or not the function failed. Pass both functions the original pointer
from vollo-rt, not a copy of the message or an error from vollo-cfg. The Rust
wrapper exposes `Error::code()`, `Error::is_would_block()` and `Error::is_timeout()`,
and frees errors on drop.

## Initialisation

A vollo context is created by calling `vollo_rt_init`.
Add an accelerator by using the `vollo_rt_add_accelerator` function, or
`vollo_rt_add_device` to name a card by its PCI address (see [hosts with more
than one accelerator](accelerator-setup.md#hosts-with-more-than-one-accelerator)).

```c
/**
 * Initialise the vollo-rt context. This must be called before any other vollo-rt functions.
 *
 * Logging level can be configured by setting the environment variable `VOLLO_RT_LOG` (or
 * `RUST_LOG`, which it takes precedence over) to one of: "error", "warn", "info", "debug", or
 * "trace"
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
 * Add an accelerator using a device specifier string. This is a more versatile version of
 * `vollo_rt_add_accelerator` that takes a string instead of just an index.
 *
 * `accelerator_index` assigns the logical accelerator slot used by query APIs such as
 * `vollo_rt_accelerator_num_cores` and `vollo_rt_accelerator_block_size`.
 *
 * - `0` -- vollo accelerator 0
 * - `0.3` -- VF 3 of vollo accelerator 0, numbered from 0 as `vollo-tool list` shows
 * - `01:00.0` -- vollo accelerator with PCI address 01:00.0
 * - `vm:vollo` -- anonymous Vollo VM, bit-accurate by default
 * - `vm+f32:vollo` -- anonymous Vollo VM in non-bit-accurate mode
 * - `vm:bittware-ia420f-c6b32` -- Vollo VM constrained to the given named hw config preset
 * - `vm+bit-accurate:bittware-ia420f-c6b32` -- same named VM in bit-accurate mode (the default)
 * - `vm:path/to/hw-config.json` -- Vollo VM constrained to the given hw config JSON file
 * - `vm+f32:path/to/hw-config.json` -- same JSON-configured VM in non-bit-accurate mode
 *
 * This should be called after `vollo_rt_init` but before `vollo_rt_load_program`.
 */
vollo_rt_error_t vollo_rt_add_device(
  vollo_rt_context_t vollo, size_t accelerator_index, const char* device_spec);
```

## Using a virtual function

On a V80, V80LL or Artena card split into virtual functions (VFs), each thread or process that
drives the card creates its own context and opens its own VF with `vollo_rt_add_device`, naming it as `<card>.<vf>` (such as `"0.3"`) or by its PCI address. It
then loads the `.vollo-vf` virtual program that `vollo-tool setup-vfs` wrote for that VF with
`vollo_rt_load_program`, in place of the `.vollo` program. After that, the API works as it does on
a whole card. See [Configuring the Accelerator for Multiple Processes](multiprocess.md).

## Loading a program

A program is loaded onto the Vollo accelerator using the `vollo_rt_load_program` function.

```c
/**
 * Load a program onto the Vollo accelerators.
 * This should be called after `vollo_rt_add_accelerator`
 *
 * A Vollo program is generated by the Vollo compiler, it is typically named
 * "<program_name>.vollo".
 * The program is intended for a specific hardware config (number of accelerators,
 * cores and other configuration options), this function will return an
 * error if any accelerator configuration is incompatible with the program.
 * Once loaded, the program provides inference for several models concurrently.
 *
 * Note: This should only be called once per `vollo_rt_context_t`, as such if
 * a program needs to be changed or reset, first `vollo_rt_destroy` the current
 * context, then start a new context with `vollo_rt_init`.
 */
vollo_rt_error_t vollo_rt_load_program(vollo_rt_context_t vollo, const char* program_path);
```

## Model metadata

Once a program is loaded, it provides inference for one or more models. Metadata about
a model is obtained with `vollo_rt_model_*` functions.

Each model can have multiple distinct inputs and outputs. Each input and each output has
a multi-dimensional shape associated with it. All of the metadata is defined by the program
as supplied by the Vollo compiler. All the shapes are statically defined.

Some models can be compiled as streaming statefully over a dimension, that dimension is then
erased from the inference shape but its possition can be recovered in the model metadata.

```c
/**
 * Inspect the number of models in the program loaded onto the vollo.
 *
 * Programs can contain multiple models, a `model_index` is used to select a
 * specific model
 */
size_t vollo_rt_num_models(vollo_rt_context_t vollo);

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
 * The return value is an array of dims containing the input shape
 * Use `vollo_rt_model_input_shape_len` to get the number of axes in the shape.
 *
 * For backwards compatibility the array is also 0-terminated, but that should not be relied upon
 * in order to correctly support shapes containing a 0 dimension
 *
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
 * Get the number of axes in the shape for the input at a given index
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 * - `input_index < vollo_rt_model_num_inputs`
 */
size_t vollo_rt_model_input_shape_len(
  vollo_rt_context_t vollo, size_t model_index, size_t input_index);

/**
 * Get the shape for output at a given index
 *
 * The return value is an array of dims containing the output shape
 * Use `vollo_rt_model_output_shape_len` to get the number of axes in the shape.
 *
 * For backwards compatibility the array is also 0-terminated, but that should not be relied upon
 * in order to correctly support shapes containing a 0 dimension
 *
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
 * Get the number of axes in the shape for the output at a given index
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 * - `output_index < vollo_rt_model_num_outputs`
 */
size_t vollo_rt_model_output_shape_len(
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
 * Get the number type of an input at the given index
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 * - `input_index < vollo_rt_model_num_inputs`
 */
number_format vollo_rt_model_input_format(
  vollo_rt_context_t vollo, size_t model_index, size_t input_index);

/**
 * Get the number type of an output at the given index
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 * - `output_index < vollo_rt_model_num_outputs`
 */
number_format vollo_rt_model_output_format(
  vollo_rt_context_t vollo, size_t model_index, size_t output_index);

/**
 * Returns whether the model can be reset to its initial state (i.e. was compiled with
 * `generate_state_reset = True`)
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 */
bool vollo_rt_model_is_resettable(vollo_rt_context_t vollo, size_t model_index);
```

## Running inference

The interface returns results asynchronously so that inference requests can be made as fast
as the system can support, without blocking on output data being returned. This way, it also
supports running multiple requests concurrently.
Before any compute is started a job with associated input and output buffers needs to be
registered with the runtime using one of `vollo_rt_add_job_bf16` or `vollo_rt_add_job_fp32`.

The `bf16` variant uses [`bfloat16`](https://cloud.google.com/tpu/docs/bfloat16)
which is effectively a cropped version of single precision floating point format
`fp32` (same exponent, smaller mantissa).
Note: do NOT use C floating point literals for `bf16` as it is simply a `uint16_t` in the API

A `fp32` variant is also provided despite the Vollo accelerator expecting its
inputs and outputs to be in `fp16`. If you are working with `fp32`, prefer
this version instead of the `bf16` variant as it is able to make the conversion
while copying to/from DMA buffers, avoiding an extra copy.

```c
/**
 * Sets up a computation on the vollo accelerator where the inputs and outputs are in brain-float 16
 * format.
 *
 * Note: The computation may be started by this call, or only on the next call to vollo_rt_poll:
 * either way it is possible to set up several computations that are kicked off at the same time.
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

vollo_rt_error_t vollo_rt_add_job_fp32(
  vollo_rt_context_t vollo,
  size_t model_index,
  uint64_t user_ctx,
  const float* const* input_data,
  float* const* output_data);
```

To actually start and later complete an inference you must use the `vollo_rt_poll` function
multiple times. It is typically called in a loop (with a timeout) until some or all of the jobs
are completed.

```c
/**
 * Poll the vollo accelerator for completion.
 *
 * Note: Polling also initiates transfers for new jobs, so poll must be called
 * before any progress on these new jobs can be made.
 *
 *   num_completed: out: the number of completed user_ctx returned
 *   returned_user_ctx: buffer for the returned user_ctx of completed jobs, this will only be
 *                      valid until the next call to vollo_rt_poll.
 */
vollo_rt_error_t vollo_rt_poll(
  vollo_rt_context_t vollo, size_t* num_completed, const uint64_t** returned_user_ctx);
```

## Partial updates

To reduce the amount of data that needs to be transferred to/from the accelerator, is is possible to
update specific indices of the input buffer stored on the accelerator.

There are two functions to achieve this:

- a simple one for a single bf16 input: `vollo_rt_add_job_bf16_partial_update`
- a more general one for multiple inputs/outputs and fp32 data: `vollo_rt_add_job_partial_update`

```c
/**
 * Sets up a computation on the vollo accelerator where the inputs and outputs are in brain-float 16
 * format.
 *
 * Takes the input from the previous job and updates individual values as provided and uses that as
 * the new input. This can be more efficient due to smaller IO requirements.
 *
 * Limitations:
 * - Only single model programs are supported
 * - Only single input models are supported
 * - A full update must be queued before the first partial update
 * - An update may set at most one value per input element, and at most 65528; a full update
 *   has no such limit
 *
 * Note: The computation may be started by this call, or only on the next call to vollo_rt_poll:
 * either way it is possible to set up several computations that are kicked off at the same time.
 *
 * - vollo:
 *     the context that the computation should be run on
 * - model_index:
 *     the model to run
 * - user_ctx:
 *     a user context that will be returned on completion
 * - num_input_updates:
 *     The number of elements in the input array to be updated
 *     It MUST be at most the number of input elements (see `vollo_rt_model_input_num_elements`),
 *     although using `vollo_rt_add_job_bf16` will be more efficient when updating many elements
 * - input_update_indices:
 *     An array of indices (with `num_input_updates` elements) of the elements to update
 *     Each index MUST be less than the number of input elements
 *     (see `vollo_rt_model_input_num_elements`)
 *     Updating multiple times the same index in a given update has undefined semantics
 *     lifetime:
 *       - The input_update_indices array needs to live until `vollo_rt_poll` returns with the
 *         completion for this job
 * - input_update_values:
 *     An array of values (with `num_input_updates` elements) with the new values of the
 *     elements to update
 *     Values may not be NaN
 *     lifetime:
 *       - The input_update_values array needs to live until `vollo_rt_poll` returns with the
 *         completion for this job
 * - output_data:
 *     a pointer to the start of an array with pointers to the start of the data to each output
 *     buffer the number of outputs is given by `vollo_rt_model_num_outputs` each output length is
 *     the product of the shape given by `vollo_rt_model_output_shape`
 *     (or more convenient: `vollo_rt_model_output_num_elements`)
 *     lifetime:
 *       - The outer array only needs to live until `vollo_rt_add_job_bf16_partial_update` returns
 *       - The output buffers need to live until `vollo_rt_poll` returns with the completion for
 *         this job
 */
vollo_rt_error_t vollo_rt_add_job_bf16_partial_update(
  vollo_rt_context_t vollo,
  size_t model_index,
  uint64_t user_ctx,
  uint32_t num_input_updates,
  const uint32_t* input_update_indices,
  const bf16* input_update_values,
  bf16* const* output_data);
```

## Resetting model state

Stateful models can have their state reset with `vollo_rt_add_reset_job`. The model must have been
compiled with `generate_state_reset = True`.

```c
/**
 * Resets a model to its initial state.
 *
 * Unlike `vollo_rt_add_job`, this function does not take in a `user_ctx`, because there is no
 * output to receive. Consequently, `vollo_rt_poll` will not notify you when a reset job is
 * done, but it will be done before any subsequent jobs for that model.
 *
 * Note: The reset is only started on the next call to `vollo_rt_poll`. This way it is possible
 * to set up several computations that are kicked off at the same time. Reset jobs are in the
 * same queue as regular jobs.
 *
 * - vollo:
 *     the context that the model should be reset on
 * - model_index:
 *     the model to reset
 *
 * Requirements (panics otherwise):
 * - a program was loaded with `vollo_rt_load_program`
 * - `model_index < vollo_rt_num_models`
 * - the model was compiled with `generate_state_reset = True`
 */
vollo_rt_error_t vollo_rt_add_reset_job(vollo_rt_context_t vollo, size_t model_index);
```

## External input (experimental)

An external-input job's inputs are written straight into the card's input
aperture by a producer outside the runtime, such as a GPU, or by CPU threads
through `vollo_rt_write_input_cpu`. Its outputs come back through
`vollo_rt_poll` like any other job's. It works with a card over PCIe, either
the whole card or a virtual function (see
[Using a virtual function](#using-a-virtual-function)), including in a virtual
machine. It does not yet work with the Vollo VM, the software simulator.

The API is experimental: the functions may change or be removed, and the PCIe
writes the card accepts may change, in a later release.

No other drivers are needed for the card or the GPU, and the runtime does not
use CUDA. The GPU writes straight to the card, not through host memory.
Processes without `CAP_SYS_ADMIN` need one NVIDIA driver setting (see
[Registering the aperture with NVIDIA drivers](#registering-the-aperture-with-nvidia-drivers)).
Whether peer writes work depends on the PCIe topology, ACS, IOMMU and GPU
driver, so check that the GPU's writes reach the card correctly on the target
platform before relying on them.

The SDK's `example/gpu` directory has a complete CUDA example, `gpu_input.cu`,
with its host setup in the README there.

### Adding a job

1. Get each input's format and size as for any job, with
   `vollo_rt_model_input_format` and `vollo_rt_model_input_num_elements`, and
   its padded size with `vollo_rt_model_input_transfer_size`.
2. For a GPU producer, get the aperture's address and size with
   `vollo_rt_get_external_input_aperture` and register it with the GPU driver.
   For CUDA, use `cuMemHostRegister` with
   `CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP`, then
   `cuMemHostGetDevicePointer` (with the runtime API, `cudaHostRegister` with
   `cudaHostRegisterIoMemory | cudaHostRegisterMapped`, then
   `cudaHostGetDevicePointer`, as the example does). Register once after loading
   the program; the registration can stay while ordinary jobs are added too.
   CPU-only use needs neither step.
3. Add the job with `vollo_rt_add_external_input_job`, giving the producer's
   address for the start of the aperture (zero for CPU-only use) and an array
   with room for one `vollo_rt_external_input_t` per model input. The job is
   submitted at once, in order with the context's other jobs, and its whole
   padded input is reserved in the card's input ring. The descriptors are
   opaque: `vollo_rt_external_input_producer_address` gives an input's producer
   address, and its format and sizes are those from step 1. An input's
   destination is contiguous even where it wraps around the ring.
4. Write each input exactly once, by one producer: the GPU, writing the input
   in its format followed by zeros up to its padded size, or
   `vollo_rt_write_input_cpu`, which converts from BF16 or FP32 and pads. Inputs
   can be written in any order. Empty inputs need no writes.
5. Poll as usual. The job completes once all of its input has arrived; until
   then, no job added after it completes. No acknowledgement from the producer is
   needed.

`vollo_rt_error_would_block` means the job could not be added yet: the input
ring is short of space, the maximum number of jobs is already in flight, or
earlier jobs added to this context are not yet submitted. Nothing was submitted
or kept, and the descriptors are unchanged. Poll, then add the job again:
polling submits queued jobs and returns completed ones.
A job whose padded input (the sum of its inputs' padded sizes) is larger than
`vollo_rt_external_input_max_input_bytes`, the context's share of the ring, is
an ordinary error.

```c
vollo_rt_external_input_t inputs[NUM_MODEL_INPUTS];
vollo_rt_error_t err = vollo_rt_add_external_input_job(
    vollo, model_index, user_ctx, output_formats, output_buffers, producer_aperture, inputs);
if (vollo_rt_error_code(err) == vollo_rt_error_would_block) {
    vollo_rt_destroy_err(err);
    /* Poll, then add the job again. */
} else if (err != NULL) {
    fprintf(stderr, "%s\n", err);
    vollo_rt_destroy_err(err);
} else {
    /* Write each input through its producer address or vollo_rt_write_input_cpu. */
}
```

A job whose input never arrives does not complete; there is no way to cancel
it. While some of its input has not arrived, `vollo_rt_poll` returns
`vollo_rt_error_timeout` each time it has waited a while without that input
(see [Errors](#errors)).

### Writing input from the CPU

**A CPU must write the aperture with `vollo_rt_write_input_cpu`, never with
`memcpy` or other stores, even if they appear to work.** The aperture is a
device port, not memory: the card accepts only whole, aligned 16-byte lanes,
each written exactly once, and ordinary stores can be split, merged or repeated
by the compiler or CPU. The function makes stores the card accepts. It takes a
descriptor, the source's format (BF16 or FP32) and a source holding the whole
input, aligned for its format and outside the aperture. On return the source
may be reused.

The function does not use the context, so worker threads can write inputs while
the context's thread adds and polls jobs. If the context's thread reuses the
`inputs` array for later jobs, give each worker a copy of its descriptor rather
than a pointer into the array. A descriptor is valid until its job completes.

| Operations | May overlap? |
| --- | --- |
| Writes of different inputs, of the same job or of different jobs | Yes |
| A CPU write and the context's thread adding or polling jobs | Yes |
| A CPU write and a GPU write of a different input | Yes |
| Two writes of the same input, even one after the other | No |
| Two calls that use the same context | No |
| Any write and program reload or `vollo_rt_destroy` | No |

Program reload and `vollo_rt_destroy` end the aperture's mapping. Before either,
stop every writer, including workers that have been given a descriptor but not
yet called the function, and unregister the aperture from the GPU driver.
`vollo_rt_destroy` does not synchronise GPU streams.

### GPU writes

The aperture accepts these PCIe writes:

- Each write is a whole number of aligned 16-byte lanes, with every byte enabled.
- A write is at most `vollo_rt_external_input_max_write_bytes` and does not
  cross a multiple of `vollo_rt_external_input_write_boundary`. These depend on
  the bitstream and the card's PCIe settings: either the card's PCIe Max Payload
  Size (at most 2048 bytes) and 4 KiB, or 64 bytes for both. Set the GPU's Max
  Payload Size no larger than `vollo_rt_external_input_max_write_bytes`; under
  different root ports the two can differ, and Linux's `pci=pcie_bus_peer2peer`
  boot option sets every device to 128 bytes.
- A write that spans more than one 64-byte word starts on a word.
- Each 16-byte lane of the input, padding included, is written exactly once, in
  any order: the card takes a 64-byte word once all four of its lanes have
  arrived. Nothing is read, nothing outside the job's inputs is written, and an
  input is not touched after its last write.

These rules are about the writes that reach the card, not the GPU's store
instructions. No CUDA operation (a vector store, a warp's stores or
`cuMemcpyAsync`) is guaranteed to produce writes that meet them, and
synchronising a stream later cannot repair a lane written twice or a write the
card refused. Check what the GPU sends on the
target platform.

A GPU may merge stores, including stores from different kernels, into one
write. A store that leaves a 64-byte word partly written can then be merged
with the next word's store into a write that starts inside a word and crosses
into the next, which the card refuses. Write each 64-byte word whole with one
instruction, for example as 16-byte stores from four adjacent threads of a warp,
not as separate partial-word stores. Since Volta a warp's threads can diverge and
store separately, so compute every thread's value without branching on the
thread, call `__syncwarp()` just before the store, and make the store volatile so
the compiler neither splits nor merges it. The example's kernel does all of
this.

```cpp
void* host_address;
size_t size_bytes;
check_vollo(vollo_rt_get_external_input_aperture(context, &host_address, &size_bytes));
check_cuda(cuMemHostRegister(host_address, size_bytes,
    CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP));
CUdeviceptr aperture;
check_cuda(cuMemHostGetDevicePointer(&aperture, host_address, 0));

// A model with two inputs: the GPU writes one, the CPU the other.
number_format format0 = vollo_rt_model_input_format(context, model, 0);
size_t elements0 = vollo_rt_model_input_num_elements(context, model, 0);
size_t transfer0 = vollo_rt_model_input_transfer_size(context, model, 0);
vollo_rt_external_input_t inputs[2];
check_vollo(vollo_rt_add_external_input_job(context, model, user_ctx,
    output_formats, host_outputs, aperture, inputs));
launch_producer(stream, vollo_rt_external_input_producer_address(&inputs[0]),
    format0, elements0, transfer0);
check_vollo(vollo_rt_write_input_cpu(&inputs[1], number_format_fp32, cpu_input_1));
poll_until_completed(context, user_ctx);

// Only when the aperture is no longer needed:
check_cuda(cuStreamSynchronize(stream));
check_cuda(cuMemHostUnregister(host_address));
```

### Registering the aperture with NVIDIA drivers

NVIDIA drivers register I/O memory only for processes with `CAP_SYS_ADMIN`. For
any other user, `cuMemHostRegister` with `CU_MEMHOSTREGISTER_IOMEMORY` fails with
`CUDA_ERROR_NOT_PERMITTED`
(`cudaErrorNotPermitted` from the runtime API). To let other users register the aperture, set the
driver's `PeerMappingOverride` registry key, rebuild the initramfs so that the
key also applies at boot, and reload the driver:

```sh
echo 'options nvidia NVreg_RegistryDwords="PeerMappingOverride=1"' |
    sudo tee /etc/modprobe.d/nvidia-peermapping.conf
# Ubuntu, among others, loads the driver from the initramfs at boot, with the
# initramfs's own copy of /etc/modprobe.d.
sudo update-initramfs -u  # Debian and Ubuntu; sudo dracut -f on Fedora and RHEL
# Reloading needs a GPU with no users, including a display server.
sudo systemctl stop nvidia-persistenced  # if it is running
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe -a nvidia nvidia_modeset nvidia_drm nvidia_uvm
sudo systemctl start nvidia-persistenced # if it was running
grep RegistryDwords /proc/driver/nvidia/params  # shows PeerMappingOverride=1
```

If another file already sets `NVreg_RegistryDwords`, add the key to that value
instead, separated by a semicolon. The setting applies to every user of the GPU
on the host. The driver registers only ranges the calling process has already
mapped as I/O memory, so a process can direct GPU writes only at device memory
it has mapped itself, such as the Vollo aperture it opened.
