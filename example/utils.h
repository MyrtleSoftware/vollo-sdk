#ifndef UTILS_HEADER
#define UTILS_HEADER

#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */

#include <time.h>
#include <vollo-rt.h>

// Helper to exit when an error is encountered
#define EXIT_ON_ERROR(expr)                 \
  do {                                      \
    vollo_rt_error_t _err = (expr);         \
    if (_err != NULL) {                     \
      fprintf(stderr, "error: %s\n", _err); \
      vollo_rt_destroy_err(_err);           \
      exit(EXIT_FAILURE);                   \
    }                                       \
  } while (0)

#define NANOSECONDS (1000 * 1000 * 1000)

double diff_timespec_ns(struct timespec from, struct timespec to);
long long diff_timespec_ns_ll(struct timespec from, struct timespec to);

// Compare two doubles
int compare_double(const void* a, const void* b);

// Convert from bf16 to float
float bf16_to_float(bf16 x);

// Convert from float to bf16
// This conversion truncates the mantissa instead of rounding
bf16 float_to_bf16(float x);

// Generate a random float in the range ± 1.0
float rand_float();

// Generate a random bf16 in the range ± 1.0
bf16 rand_bf16();

// Partially shuffle an array
void partial_rand_shuffle(uint32_t partial_count, size_t len, uint32_t* elems);

typedef struct {
  double mean_latency_ns;
  double best_latency_ns;
  double median_latency_ns;
  double p99_latency_ns;
  double worst_latency_ns;
} latency_summary;

latency_summary summarize_latencies(size_t len, double* latencies);

#endif  // UTILS_HEADER
