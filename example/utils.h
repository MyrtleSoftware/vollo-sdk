#pragma once

#include <assert.h>
#include <string.h>
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

#define NANOSECONDS (1000 * 1000 * 1000)

__attribute__((unused)) static double diff_timespec_ns(struct timespec from, struct timespec to) {
  return (double)((long long)(to.tv_sec - from.tv_sec) * NANOSECONDS + (to.tv_nsec - from.tv_nsec));
}

// Compare two doubles
__attribute__((unused)) static int compare_double(const void* a, const void* b) {
  if (*(double*)a > *(double*)b)
    return 1;
  else if (*(double*)a < *(double*)b)
    return -1;
  else
    return 0;
}

// Convert from bf16 to float
__attribute__((unused)) static float bf16_to_float(bf16 x) {
  uint32_t y = ((uint32_t)x) << 16;
  float y_float;
  memcpy(&y_float, &y, sizeof(float));
  return y_float;
}

// Convert from float to bf16
// This conversion truncates the mantissa instead of rounding
__attribute__((unused)) static bf16 float_to_bf16(float x) {
  uint32_t x_int;
  memcpy(&x_int, &x, sizeof(float));
  return (bf16)(x_int >> 16);
}

// Generate a random float in the range ± 1.0
__attribute__((unused)) static float rand_float() {
  return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

// Generate a random bf16 in the range ± 1.0
__attribute__((unused)) static bf16 rand_bf16() {
  float x = rand_float();
  uint32_t x_int;
  memcpy(&x_int, &x, sizeof(uint32_t));
  return (bf16)(x_int >> 16);
}

// Partially shuffle an array
__attribute__((unused)) static void partial_rand_shuffle(
  uint32_t partial_count, size_t len, uint32_t* elems) {
  assert(partial_count <= len);

  for (uint32_t i = 0; i < partial_count; i++) {
    // randomly select an index in the rest of the array
    // Note: this is not uniform, but good enough for this example
    uint32_t n = i + (uint32_t)rand() % ((uint32_t)len - i);

    // swap with the current index
    uint32_t t = elems[i];
    elems[i] = elems[n];
    elems[n] = t;
  }
}

typedef struct {
  double mean_latency_ns;
  double best_latency_ns;
  double median_latency_ns;
  double p99_latency_ns;
  double worst_latency_ns;
} latency_summary;

__attribute__((unused)) static latency_summary summarize_latencies(size_t len, double* latencies) {
  qsort(latencies, len, sizeof(double), compare_double);

  double sum_latencies_ns = 0.0;
  for (size_t i = 0; i < len; i++) {
    sum_latencies_ns += latencies[i];
  }

  return (latency_summary){
    .mean_latency_ns = sum_latencies_ns / (double)len,
    .best_latency_ns = latencies[0],
    .median_latency_ns = latencies[len / 2],
    .p99_latency_ns = latencies[(99 * len) / 100],
    .worst_latency_ns = latencies[len - 1],
  };
}
