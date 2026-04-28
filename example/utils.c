#include "utils.h"

#include <errno.h>
#include <string.h>
#include <vollo-rt.h>

double diff_timespec_ns(struct timespec from, struct timespec to) {
  return (double)((long long)(to.tv_sec - from.tv_sec) * NANOSECONDS + (to.tv_nsec - from.tv_nsec));
}

long long diff_timespec_ns_ll(struct timespec from, struct timespec to) {
  return (long long)(to.tv_sec - from.tv_sec) * NANOSECONDS + (to.tv_nsec - from.tv_nsec);
}

// Compare two doubles
int compare_double(const void* a, const void* b) {
  if (*(double*)a > *(double*)b)
    return 1;
  else if (*(double*)a < *(double*)b)
    return -1;
  else
    return 0;
}

// Convert from bf16 to float
float bf16_to_float(bf16 x) {
  uint32_t y = ((uint32_t)x) << 16;
  float y_float;
  memcpy(&y_float, &y, sizeof(float));
  return y_float;
}

// Convert from float to bf16
// This conversion truncates the mantissa instead of rounding
bf16 float_to_bf16(float x) {
  uint32_t x_int;
  memcpy(&x_int, &x, sizeof(float));
  return (bf16)(x_int >> 16);
}

// Generate a random float in the range ± 1.0
float rand_float(void) {
  return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

// Generate a random bf16 in the range ± 1.0
bf16 rand_bf16(void) {
  float x = rand_float();
  uint32_t x_int;
  memcpy(&x_int, &x, sizeof(uint32_t));
  return (bf16)(x_int >> 16);
}

// Partially shuffle an array
void partial_rand_shuffle(size_t partial_count, size_t len, uint32_t* elems) {
  ALWAYS_ASSERT(partial_count <= len);

  for (size_t i = 0; i < partial_count; i++) {
    // randomly select an index in the rest of the array
    // Note: this is not uniform, but good enough for this example
    size_t n = i + (size_t)rand() % (len - i);

    // swap with the current index
    uint32_t t = elems[i];
    elems[i] = elems[n];
    elems[n] = t;
  }
}

size_t parse_size_arg(const char* s, const char* opt_name) {
  char* end;
  errno = 0;
  unsigned long val = strtoul(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') {
    fprintf(stderr, "invalid value for %s: '%s'\n", opt_name, s);
    exit(EXIT_FAILURE);
  }
  return (size_t)val;
}

long parse_long_arg(const char* s, const char* opt_name) {
  char* end;
  errno = 0;
  long val = strtol(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') {
    fprintf(stderr, "invalid value for %s: '%s'\n", opt_name, s);
    exit(EXIT_FAILURE);
  }
  return val;
}

latency_summary summarize_latencies(size_t len, double* latencies) {
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
