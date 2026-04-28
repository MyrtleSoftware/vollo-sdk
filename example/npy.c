// NOTE: this implementation assumes a little-endian host. The NPY dtype
// descriptor is hard-coded to '<f4' (little-endian float32) and the float
// payload is read from / written to disk without any byte swapping, so on a
// big-endian host the file contents would be misinterpreted. Vollo only
// targets little-endian architectures (x86_64) so this
// is sufficient in practice.

// utils.h transitively pulls in <time.h> via its own declarations, which
// require a POSIX feature test macro on older glibc for struct timespec to
// be visible. Match the guard used in example.c / partial_update.c.
#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */

#include "npy.h"

#include "utils.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void free_npy(NpyArray array) {
  free(array.buffer);
}

// Read a NPY file
// Only dtype float32 supported
NpyArray read_npy(const char* file_path) {
  NpyArray array;
  size_t num_elements = 1;

  FILE* fptr = fopen(file_path, "rb");

  if (fptr == NULL) {
    fprintf(stderr, "Could not open %s, error: %s\n", file_path, strerror(errno));
    exit(EXIT_FAILURE);
  }

  unsigned char init_header[10];
  size_t bytes;

  bytes = fread(init_header, 1, 10, fptr);

  if (bytes != 10) {
    fprintf(stderr, "read_npy: Could not read header\n");
    exit(EXIT_FAILURE);
  }

  if (memcmp(init_header, "\x93NUMPY", 6) != 0) {
    fprintf(stderr, "read_npy: Not a numpy file\n");
    exit(EXIT_FAILURE);
  }

  if (memcmp(&init_header[6], "\x01\x00", 2) != 0) {
    fprintf(stderr, "read_npy: Only version 1.0 of .npy supported\n");
    exit(EXIT_FAILURE);
  }

  uint16_t header_len
    = (uint16_t)(((uint16_t)init_header[8]) | (uint16_t)(((uint16_t)init_header[9]) << 8));
  // The shortest possible valid NPY header dict is longer than 10 bytes; anything
  // shorter can never be parsed successfully and indicates a corrupt file.
  ALWAYS_ASSERT(header_len >= 10);
  char* header = (char*)malloc(sizeof(char) * header_len);
  if (header == NULL) {
    fprintf(stderr, "read_npy: Could not allocate header buffer\n");
    exit(EXIT_FAILURE);
  }
  bytes = fread(header, 1, header_len, fptr);

  if (bytes != header_len) {
    fprintf(stderr, "read_npy: Could not read header (descr)\n");
    exit(EXIT_FAILURE);
  }

  size_t offset = 0;
#define SKIP_SPACES()                                         \
  do {                                                        \
    while (offset < header_len && header[offset] == '\x20') { \
      offset++;                                               \
    }                                                         \
  } while (0)
#define TRY(expected, len) \
  (offset + len <= header_len && memcmp(&header[offset], expected, len) == 0)
#define MATCH_PATTERN(expected, len, msg)                               \
  do {                                                                  \
    if (TRY(expected, len)) {                                           \
      offset += len;                                                    \
    } else {                                                            \
      fprintf(stderr, "read_npy: Error while reading header%s\n", msg); \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

#define MATCH_CHAR(c) MATCH_PATTERN(c, 1, " (expected " c ")")

  SKIP_SPACES();
  MATCH_CHAR("{");
  SKIP_SPACES();

  bool parse_descr = false;
  bool parse_fortran_order = false;
  bool parse_shape = false;

  while (true) {
    if (!parse_descr && TRY("'descr'", 7)) {
      parse_descr = true;
      offset += 7;

      SKIP_SPACES();
      MATCH_CHAR(":");
      SKIP_SPACES();
      MATCH_PATTERN("'<f4'", 5, " (only float32 supported)");
      SKIP_SPACES();

    } else if (!parse_fortran_order && TRY("'fortran_order'", 15)) {
      parse_fortran_order = true;
      offset += 15;

      SKIP_SPACES();
      MATCH_CHAR(":");
      SKIP_SPACES();
      MATCH_PATTERN("False", 5, " (only C order supported)");
      SKIP_SPACES();

    } else if (!parse_shape && TRY("'shape'", 7)) {
      parse_shape = true;
      offset += 7;
      SKIP_SPACES();
      MATCH_CHAR(":");
      SKIP_SPACES();
      MATCH_CHAR("(");
      SKIP_SPACES();

      array.shape_len = 0;

      while (!TRY(")", 1)) {
        ALWAYS_ASSERT(array.shape_len < NPY_MAX_SHAPE_LEN);

        char* start = &header[offset];
        char* end = start;
        errno = 0;
        size_t x = strtoull(start, &end, 10);

        if (start == end) {
          fprintf(stderr, "read_npy: Couldn't parse shape\n");
          exit(EXIT_FAILURE);
        }

        if (errno == ERANGE) {
          fprintf(stderr, "read_npy: shape dim out of range\n");
          exit(EXIT_FAILURE);
        }

        if (x != 0 && num_elements > SIZE_MAX / x) {
          fprintf(stderr, "read_npy: shape would overflow size_t\n");
          exit(EXIT_FAILURE);
        }
        num_elements *= x;

        array.shape[array.shape_len] = x;
        array.shape_len++;

        offset += (size_t)(end - start);
        SKIP_SPACES();

        if (TRY(",", 1)) {
          offset++;
          SKIP_SPACES();
        } else {
          // No comma, it must be the end of shape
          break;
        }
      }

      MATCH_CHAR(")");
      SKIP_SPACES();

    } else {
      fprintf(stderr, "read_npy: Unexpected field in header");
      exit(EXIT_FAILURE);
    }

    if (parse_descr && parse_fortran_order && parse_shape) {
      break;
    }

    MATCH_CHAR(",");
    SKIP_SPACES();
  }

  if (TRY(",", 1)) {
    offset++;
  }
  SKIP_SPACES();
  MATCH_CHAR("}");
  SKIP_SPACES();
  MATCH_CHAR("\n");
  if (offset != header_len) {
    fprintf(stderr, "read_npy: Unexpected end of header");
    exit(EXIT_FAILURE);
  }

#undef MATCH_CHAR
#undef MATCH_PATTERN
#undef TRY
#undef SKIP_SPACES

  array.buffer_len = num_elements;

  if (num_elements > 0) {
    array.buffer = (float*)malloc(sizeof(float) * num_elements);
    if (array.buffer == NULL) {
      fprintf(stderr, "read_npy: Could not allocate array buffer\n");
      exit(EXIT_FAILURE);
    }

    bytes = fread(array.buffer, sizeof(float), num_elements, fptr);

    if (bytes != num_elements) {
      fprintf(stderr, "read_npy: Could not read data\n");
      exit(EXIT_FAILURE);
    }
  } else {
    array.buffer = NULL;
  }

  free(header);
  ALWAYS_ASSERT(fclose(fptr) == 0);

  return array;
}

// Write a NPY file
void write_npy(const char* file_path, NpyArray array) {
  FILE* fptr = fopen(file_path, "wb");

  if (fptr == NULL) {
    fprintf(stderr, "Could not open %s, error: %s\n", file_path, strerror(errno));
    exit(EXIT_FAILURE);
  }

  // Multiple of 64 bytes
  char HEADER[128];
  memset(HEADER, '\x20', sizeof(HEADER));  // Padding with ASCII spaces
  memcpy(
    HEADER,
    "\x93NUMPY\x01\x00\x76\x00",
    6 + 2 + 2);  // magic string + version + len
                 // 128 - 6 - 2 - 2 = 118 = 0x76 = [0x76, 0x00] in little endian

  size_t offset = 6 + 2 + 2;
  int ret;
#define WRITE_NPY_CHUNK(...)                                               \
  do {                                                                     \
    ret = snprintf(&HEADER[offset], sizeof(HEADER) - offset, __VA_ARGS__); \
    if (ret < 0 || offset + (size_t)ret >= sizeof(HEADER)) {               \
      fprintf(stderr, "write_npy: Error creating header\n");               \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
    offset += (size_t)ret;                                                 \
  } while (0)

  WRITE_NPY_CHUNK("{'descr': '<f4', 'fortran_order': False, 'shape': (");

  size_t num_elems = 1;

  if (array.shape_len > 0) {
    WRITE_NPY_CHUNK("%zu", array.shape[0]);
    num_elems *= array.shape[0];

    if (array.shape_len == 1) {
      // Add trailing comma for singleton tuples
      WRITE_NPY_CHUNK(",");
    }

    for (uint8_t i = 1; i < array.shape_len; i++) {
      WRITE_NPY_CHUNK(", %zu", array.shape[i]);
      num_elems *= array.shape[i];
    }
  }

  WRITE_NPY_CHUNK("), }");

  if (num_elems != array.buffer_len) {
    fprintf(
      stderr,
      "write_npy: shape implies %zu elements but buffer_len is %zu\n",
      num_elems,
      array.buffer_len);
    exit(EXIT_FAILURE);
  }

#undef WRITE_NPY_CHUNK

  HEADER[offset] = '\x20';  // Replace '\0' by a space as per the spec
  HEADER[127] = '\n';       // Terminate HEADER with a newline

  ALWAYS_ASSERT(fwrite(HEADER, 1, sizeof(HEADER), fptr) == sizeof(HEADER));

  if (num_elems > 0) {
    ALWAYS_ASSERT(array.buffer != NULL);
    ALWAYS_ASSERT(fwrite(array.buffer, sizeof(float), num_elems, fptr) == num_elems);
  }

  ALWAYS_ASSERT(fclose(fptr) == 0);
}
