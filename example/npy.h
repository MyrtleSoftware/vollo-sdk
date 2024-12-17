#ifndef NPY_HEADER
#define NPY_HEADER

// This header implements serialization/deserialization of the NPY format
// It only supports version 1.0 of the format and only for a dtype of float32
//
// Using spec from https://numpy.org/devdocs/reference/generated/numpy.lib.format.html

#include <stdint.h>
#include <string.h>

#define NPY_MAX_SHAPE_LEN 16
typedef struct {
  float* buffer;
  size_t buffer_len;
  size_t shape[NPY_MAX_SHAPE_LEN];
  uint8_t shape_len;
} NpyArray;

void free_npy(NpyArray array);

// Read a NPY file
// Only dtype float32 supported
NpyArray read_npy(const char* file_path);

// Write a NPY file
void write_npy(const char* file_path, NpyArray array);

#endif  // NPY_HEADER
