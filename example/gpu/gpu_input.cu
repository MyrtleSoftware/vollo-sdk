// Copyright(C) 2026 Myrtle Software Ltd. All rights reserved.

// GPU-produced input for Vollo (experimental external-input API).
//
// A CUDA kernel writes each job's input straight into Vollo's PCIe input aperture, with no
// copy through host memory, and the outputs arrive in host memory as usual. Each GPU job is
// checked against an ordinary job given the same input from the CPU, so use a stateless model
// such as the identity programs in the parent directory.
//
// See README.md for the host setup this needs, and the "External input" section of the C API
// guide for the full contract.

#include <algorithm>
#include <cuda_runtime.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <vollo-rt.hpp>

#define EXIT_ON_VOLLO_ERROR(expr)           \
  do {                                      \
    vollo_rt_error_t _err = (expr);         \
    if (_err != NULL) {                     \
      fprintf(stderr, "error: %s\n", _err); \
      vollo_rt_destroy_err(_err);           \
      exit(EXIT_FAILURE);                   \
    }                                       \
  } while (0)

#define EXIT_ON_CUDA_ERROR(expr)                                                                  \
  do {                                                                                            \
    cudaError_t _err = (expr);                                                                    \
    if (_err != cudaSuccess) {                                                                    \
      fprintf(                                                                                    \
        stderr, "error: %s: %s (%s)\n", #expr, cudaGetErrorString(_err), cudaGetErrorName(_err)); \
      exit(EXIT_FAILURE);                                                                         \
    }                                                                                             \
  } while (0)

static const size_t MODEL = 0;
static const size_t MAX_INPUTS = 16;

// The input value of element `i` of input `input` in job `job`. Every value is a small multiple of
// 1/64, exact in both BF16 and FP32, so the CPU and GPU produce the same bits.
__host__ __device__ static float input_value(uint32_t job, uint32_t input, uint32_t i) {
  return (float)((int)((i * 17u + job * 31u + input * 7u) % 255u) - 127) / 64.0f;
}

// All ones if `keep`, else zero, to mask a value without a branch.
__device__ static uint32_t mask_if(bool keep) {
  return 0u - (uint32_t)keep;
}

// Stores one aligned 16-byte vector with exactly one instruction. The aperture is a device port,
// not memory, so the compiler must not drop, split or merge the store, as it may with an ordinary
// one; a volatile store is kept as written.
__device__ static void store_lane(uint4* dest, uint4 value) {
  asm volatile("st.volatile.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(dest), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

// Writes one complete input to its destination in the aperture: `data_bytes` of native-format data
// followed by zero padding to `transfer_bytes`. Launch it with blocks of whole warps.
//
// Each thread stores one aligned 16-byte vector, so a warp's store instruction covers 512
// contiguous bytes and writes every 64-byte input word whole, as four lanes of one instruction.
// Separate partial-word stores could be merged by the GPU into a write that crosses a word, which
// the aperture refuses. Since Volta, a warp's threads can diverge and then store separately, so
// the values are computed without branching on the thread and the warp is reconverged before the
// store. Which PCIe writes a store becomes still depends on the GPU and platform, so check them on
// yours (see README.md).
__global__ void write_input(
  uint4* dest,
  size_t transfer_bytes,
  size_t data_bytes,
  bool bf16_format,
  uint32_t job,
  uint32_t input) {
  size_t chunks = transfer_bytes / sizeof(uint4);
  size_t lane = threadIdx.x % warpSize;
  // Each warp steps through the input together: `base`, its first chunk, is the same for all its
  // threads, so they all take the loop the same number of times.
  for (size_t base = blockIdx.x * (size_t)blockDim.x + threadIdx.x - lane; base < chunks;
       base += (size_t)gridDim.x * blockDim.x) {
    size_t c = base + lane;
    uint32_t words[4];
    for (uint32_t w = 0; w < 4; w++) {
      size_t offset = c * sizeof(uint4) + w * sizeof(uint32_t);
      if (bf16_format) {
        uint32_t i = (uint32_t)(offset / 2);
        uint32_t lo = __float_as_uint(input_value(job, input, i)) >> 16;
        uint32_t hi = __float_as_uint(input_value(job, input, i + 1)) >> 16;
        words[w]
          = (lo & mask_if(offset < data_bytes)) | (hi & mask_if(offset + 2 < data_bytes)) << 16;
      } else {
        uint32_t i = (uint32_t)(offset / 4);
        words[w] = __float_as_uint(input_value(job, input, i)) & mask_if(offset < data_bytes);
      }
    }
    __syncwarp();
    // `chunks` and `base` are multiples of 4, so the four lanes of a word agree on this test and
    // each word is stored whole or not at all.
    if (c < chunks) {
      store_lane(&dest[c], make_uint4(words[0], words[1], words[2], words[3]));
    }
  }
}

// Fills `buffer` with the same input on the CPU, for the reference job.
static void host_input(
  std::vector<uint8_t>& buffer,
  number_format format,
  size_t elements,
  uint32_t job,
  uint32_t input) {
  for (uint32_t i = 0; i < elements; i++) {
    float value = input_value(job, input, i);
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if (format == number_format::number_format_bf16) {
      uint16_t half = (uint16_t)(bits >> 16);
      memcpy(&buffer[2 * i], &half, sizeof(half));
    } else {
      memcpy(&buffer[4 * i], &bits, sizeof(bits));
    }
  }
}

static size_t format_size(number_format format) {
  switch (format) {
  case number_format::number_format_bf16: return 2;
  case number_format::number_format_fp32: return 4;
  default: return 8;
  }
}

// Parses a non-negative decimal option value, or exits.
static unsigned long long parse_option(const char* name, const char* text, unsigned long long max) {
  char* end;
  errno = 0;
  unsigned long long value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || text[0] == '-' || value > max) {
    fprintf(stderr, "error: --%s: invalid value '%s'\n", name, text);
    exit(EXIT_FAILURE);
  }
  return value;
}

// Polls until the job with `user_ctx`, added after GPU job `job`, completes. A GPU job's input
// arrives asynchronously, so a poll can return vollo_rt_error_timeout while its writer is still
// queued or running; keep polling, but check the GPU stream for a failed kernel, which would never
// deliver the input.
static void wait_for(vollo_rt_context_t ctx, uint32_t job, uint64_t user_ctx, cudaStream_t stream) {
  for (unsigned timeouts_after_writer = 0;;) {
    size_t completed = 0;
    const uint64_t* user_ctxs = NULL;
    vollo_rt_error_t err = vollo_rt_poll(ctx, &completed, &user_ctxs);
    if (vollo_rt_error_code(err) == vollo_rt_error_code_t::vollo_rt_error_timeout) {
      cudaError_t gpu = cudaStreamQuery(stream);
      if (gpu != cudaSuccess && gpu != cudaErrorNotReady) {
        vollo_rt_destroy_err(err);
        fprintf(
          stderr,
          "error: GPU job %u: input kernel: %s (%s)\n",
          job,
          cudaGetErrorString(gpu),
          cudaGetErrorName(gpu));
        exit(EXIT_FAILURE);
      }
      // A busy GPU may not have run the writer yet, so wait as long as that takes. Once it has
      // finished, its writes are on their way: a second timeout means a whole poll budget has
      // passed since then without them.
      if (gpu == cudaErrorNotReady || ++timeouts_after_writer < 2) {
        vollo_rt_destroy_err(err);
        continue;
      }
      fprintf(stderr, "error: GPU job %u: %s\n", job, err);
      fprintf(
        stderr,
        "note: the input kernel finished, so its writes did not reach the card or the card "
        "refused them; see README.md\n");
      vollo_rt_destroy_err(err);
      exit(EXIT_FAILURE);
    }
    EXIT_ON_VOLLO_ERROR(err);
    for (size_t i = 0; i < completed; i++) {
      if (user_ctxs[i] == user_ctx) {
        return;
      }
    }
  }
}

int main(int argc, char** argv) {
  const char* device_spec = getenv("VOLLO_CARD_BDF") != NULL ? getenv("VOLLO_CARD_BDF") : "0";
  uint32_t num_jobs = 100;
  int cuda_device = 0;

  static struct option long_options[] = {
    {"device", required_argument, 0, 'd'},
    {"cuda-device", required_argument, 0, 'g'},
    {"num-jobs", required_argument, 0, 'n'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0},
  };
  int opt;
  while ((opt = getopt_long(argc, argv, "d:g:n:h", long_options, NULL)) != -1) {
    switch (opt) {
    case 'd': device_spec = optarg; break;
    case 'g': cuda_device = (int)parse_option("cuda-device", optarg, INT_MAX); break;
    case 'n': num_jobs = (uint32_t)parse_option("num-jobs", optarg, UINT32_MAX); break;
    case 'h':
      printf(
        "USAGE:\n    %s [--device <SPEC>] [--cuda-device <N>] [--num-jobs <N>] [PROGRAM]\n",
        argv[0]);
      printf("--device defaults to $VOLLO_CARD_BDF if set, otherwise 0\n");
      printf("PROGRAM defaults to ../identity_b<block size>.vollo\n");
      return EXIT_SUCCESS;
    default: return EXIT_FAILURE;
    }
  }

  //////////////////////////////////////////////////
  // Vollo: open the accelerator and load a program

  vollo_rt_context_t ctx;
  EXIT_ON_VOLLO_ERROR(vollo_rt_init(&ctx));
  EXIT_ON_VOLLO_ERROR(vollo_rt_add_device(ctx, 0, device_spec));

  char default_program[64];
  snprintf(
    default_program,
    sizeof(default_program),
    "../identity_b%zu.vollo",
    vollo_rt_accelerator_block_size(ctx, 0));
  const char* program = optind < argc ? argv[optind] : default_program;
  EXIT_ON_VOLLO_ERROR(vollo_rt_load_program(ctx, program));

  size_t num_inputs = vollo_rt_model_num_inputs(ctx, MODEL);
  size_t num_outputs = vollo_rt_model_num_outputs(ctx, MODEL);
  if (num_inputs > MAX_INPUTS || num_outputs > MAX_INPUTS) {
    fprintf(
      stderr, "error: the example handles models with up to %zu inputs and outputs\n", MAX_INPUTS);
    return EXIT_FAILURE;
  }

  // Each input is written in the model's format, then zeros up to its transfer size.
  struct InputLayout {
    number_format format;
    size_t data_bytes;
    size_t transfer_bytes;
  };
  InputLayout layouts[MAX_INPUTS];
  for (size_t i = 0; i < num_inputs; i++) {
    layouts[i].format = vollo_rt_model_input_format(ctx, MODEL, i);
    layouts[i].data_bytes
      = vollo_rt_model_input_num_elements(ctx, MODEL, i) * format_size(layouts[i].format);
    layouts[i].transfer_bytes = vollo_rt_model_input_transfer_size(ctx, MODEL, i);
  }

  //////////////////////////////////////////////////
  // CUDA: register Vollo's input aperture so that kernels can write to it

  EXIT_ON_CUDA_ERROR(cudaSetDevice(cuda_device));
  cudaStream_t stream;
  EXIT_ON_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  void* host_address;
  size_t size_bytes;
  EXIT_ON_VOLLO_ERROR(vollo_rt_get_external_input_aperture(ctx, &host_address, &size_bytes));
  size_t max_write_bytes = vollo_rt_external_input_max_write_bytes(ctx);
  printf(
    "Input aperture: %zu bytes, writes of up to %zu bytes within %zu-byte boundaries\n",
    size_bytes,
    max_write_bytes,
    vollo_rt_external_input_write_boundary(ctx));
  // The kernel's 512-byte warp stores need a bitstream that takes writes wider than one word.
  if (max_write_bytes <= 64) {
    fprintf(stderr, "error: this bitstream's input aperture does not take the kernel's writes\n");
    return EXIT_FAILURE;
  }

  // The aperture is device memory, not host RAM, so it needs the I/O-memory flag. Without
  // CAP_SYS_ADMIN this fails with cudaErrorNotPermitted unless the NVIDIA driver has
  // PeerMappingOverride=1; README.md gives the host setup.
  cudaError_t registered
    = cudaHostRegister(host_address, size_bytes, cudaHostRegisterIoMemory | cudaHostRegisterMapped);
  if (registered == cudaErrorNotPermitted) {
    fprintf(stderr, "error: cudaHostRegister: %s\n", cudaGetErrorString(registered));
    fprintf(stderr, "note: set the NVIDIA driver's PeerMappingOverride option; see README.md\n");
    return EXIT_FAILURE;
  }
  EXIT_ON_CUDA_ERROR(registered);
  void* aperture;
  EXIT_ON_CUDA_ERROR(cudaHostGetDevicePointer(&aperture, host_address, 0));

  // Load the kernel before publishing any job: a launch that failed after publication would leave
  // the FPGA waiting for input that never arrives.
  cudaFuncAttributes attributes;
  EXIT_ON_CUDA_ERROR(cudaFuncGetAttributes(&attributes, write_input));

  //////////////////////////////////////////////////
  // Buffers: host outputs for the GPU and reference jobs, host inputs for the reference jobs

  number_format input_formats[MAX_INPUTS];
  number_format output_formats[MAX_INPUTS];
  std::vector<std::vector<uint8_t>> host_inputs(num_inputs);
  std::vector<std::vector<uint8_t>> gpu_outputs(num_outputs), cpu_outputs(num_outputs);
  const void* input_ptrs[MAX_INPUTS];
  void* gpu_output_ptrs[MAX_INPUTS];
  void* cpu_output_ptrs[MAX_INPUTS];
  for (size_t i = 0; i < num_inputs; i++) {
    input_formats[i] = layouts[i].format;
    host_inputs[i].resize(layouts[i].data_bytes);
    input_ptrs[i] = host_inputs[i].data();
  }
  for (size_t o = 0; o < num_outputs; o++) {
    output_formats[o] = vollo_rt_model_output_format(ctx, MODEL, o);
    size_t bytes
      = vollo_rt_model_output_num_elements(ctx, MODEL, o) * format_size(output_formats[o]);
    gpu_outputs[o].resize(bytes);
    cpu_outputs[o].resize(bytes);
    gpu_output_ptrs[o] = gpu_outputs[o].data();
    cpu_output_ptrs[o] = cpu_outputs[o].data();
  }

  //////////////////////////////////////////////////
  // Run jobs

  size_t mismatches = 0;
  for (uint32_t job = 0; job < num_jobs; job++) {
    uint64_t gpu_ctx = 2 * (uint64_t)job, cpu_ctx = gpu_ctx + 1;

    // Different fill patterns, so an output that either job leaves unwritten is a mismatch.
    for (size_t o = 0; o < num_outputs; o++) {
      std::fill(gpu_outputs[o].begin(), gpu_outputs[o].end(), (uint8_t)0xff);
      std::fill(cpu_outputs[o].begin(), cpu_outputs[o].end(), (uint8_t)0x00);
    }

    // Reserve the job's input destinations. This publishes the job: the FPGA waits for its input.
    vollo_rt_external_input_t inputs[MAX_INPUTS];
    for (;;) {
      vollo_rt_error_t err = vollo_rt_add_external_input_job(
        ctx,
        MODEL,
        gpu_ctx,
        output_formats,
        gpu_output_ptrs,
        (uint64_t)(uintptr_t)aperture,
        inputs);
      if (vollo_rt_error_code(err) != vollo_rt_error_code_t::vollo_rt_error_would_block) {
        EXIT_ON_VOLLO_ERROR(err);
        break;
      }
      // The input ring or task queue is full: nothing was reserved. Poll to retire jobs and retry.
      // This example waits for every job before adding the next, so it never gets here; a
      // pipelined producer must keep the user_ctxs this poll returns.
      vollo_rt_destroy_err(err);
      size_t completed;
      const uint64_t* user_ctxs;
      EXIT_ON_VOLLO_ERROR(vollo_rt_poll(ctx, &completed, &user_ctxs));
    }

    // Write every input from the GPU, each exactly once. There is no host synchronization with
    // the GPU: the FPGA starts as soon as the input arrives.
    for (size_t i = 0; i < num_inputs; i++) {
      if (layouts[i].transfer_bytes == 0) {
        continue;
      }
      size_t chunks = layouts[i].transfer_bytes / sizeof(uint4);
      unsigned threads = 256;
      unsigned blocks = (unsigned)((chunks + threads - 1) / threads);
      write_input<<<blocks, threads, 0, stream>>>(
        (uint4*)(uintptr_t)vollo_rt_external_input_producer_address(&inputs[i]),
        layouts[i].transfer_bytes,
        layouts[i].data_bytes,
        layouts[i].format == number_format::number_format_bf16,
        job,
        (uint32_t)i);
      EXIT_ON_CUDA_ERROR(cudaGetLastError());
    }

    // The same input through an ordinary job, as the reference.
    for (size_t i = 0; i < num_inputs; i++) {
      size_t elements = layouts[i].data_bytes / format_size(layouts[i].format);
      host_input(host_inputs[i], layouts[i].format, elements, job, (uint32_t)i);
    }
    EXIT_ON_VOLLO_ERROR(vollo_rt_add_job(
      ctx, MODEL, cpu_ctx, input_formats, input_ptrs, output_formats, cpu_output_ptrs));

    // Jobs complete in submission order, so the reference job completes after the GPU job.
    wait_for(ctx, job, cpu_ctx, stream);

    bool differs = false;
    for (size_t o = 0; o < num_outputs; o++) {
      if (gpu_outputs[o] != cpu_outputs[o]) {
        if (mismatches == 0 && !differs) {
          fprintf(stderr, "job %u: output %zu differs from the reference\n", job, o);
        }
        differs = true;
      }
    }
    mismatches += differs;
  }

  //////////////////////////////////////////////////
  // Clean up: stop GPU users and unregister the aperture before destroying the context

  EXIT_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXIT_ON_CUDA_ERROR(cudaHostUnregister(host_address));
  EXIT_ON_CUDA_ERROR(cudaStreamDestroy(stream));
  vollo_rt_destroy(ctx);

  if (mismatches != 0) {
    fprintf(stderr, "%zu of %u GPU jobs differed from the reference\n", mismatches, num_jobs);
    return EXIT_FAILURE;
  }
  printf("%u GPU jobs matched the reference\n", num_jobs);
  return EXIT_SUCCESS;
}
