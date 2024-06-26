// Copyright(C) 2024 Myrtle Software Ltd. All rights reserved.

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <ostream>

/// Functions in vollo-cfg that can return an error return `vollo_cfg_error_t`.
/// NULL is returned where there are no errors, otherwise it is a null-terminated string containing
/// an error message.
///
/// Error messages are owned by vollo-cfg and can be freed with `vollo_cfg_destroy_err`
using vollo_cfg_error_t = const char*;

/// A reader is a function that executes a read on Vollo's configuration
/// bus on behalf of vollo-cfg
///
/// There are a few rules that vollo-cfg expects it to follow:
/// - All previous writes must have completed
/// - Results cannot be cached (i.e. non-prefetchable only)
/// - A single vollo-cfg read may also trigger read requests to the other aligned 4 byte words that
///   make up a 64 byte aligned word
///   i.e a vollo-cfg read of address 0x100 is allowed to trigger axi4lite reads to addresses
///   [0x100, 0x104, .. 0x138, 0x13c] (still returning the result from 0x100)
using vollo_cfg_reader_t = uint32_t (*)(void* user_context, uint32_t addr);

/// A writer is a function that executes a write on Vollo's configuration
/// bus on behalf of vollo-cfg
///
/// There are a few rules that vollo-cfg expects it to follow:
/// - Writes must not be repeated
/// - Write with an empty strobe will be ignore but must be either fully high or fully low
///   (over 4 bytes), partial strobes are not allowed
/// - Writes can be buffered but all writes must have finished before a read can be issued
using vollo_cfg_writer_t = void (*)(void* user_context, uint32_t addr, uint32_t dat);

extern "C" {

/// All APIs return the error as a c string. To prevent leaking the memory, destroy it afterwards.
void vollo_cfg_destroy_err(vollo_cfg_error_t err);

/// Print the device ID and information needed to acquire a license in JSON format to stdout.
vollo_cfg_error_t vollo_cfg_print_device_id(
  void* user_context, vollo_cfg_reader_t reader, vollo_cfg_writer_t writer);

/// Activate the license of a Vollo accelerator
///
/// Licenses are found automatically via the MYRTLE_LICENSE environment variable
///
/// The user context given here will be provided on every call to the reader and writer
///
/// Users of this library are free to use the context as they see fit in order to provide a reader
/// and writer to Vollo's configuration bus. The reads and writes will be issued from the same
/// thread as this function call.
vollo_cfg_error_t vollo_cfg_activate_license(
  void* user_context, vollo_cfg_reader_t reader, vollo_cfg_writer_t writer);

/// Load a program onto a Vollo accelerator.
///
/// A Vollo program is generated by the Vollo compiler, it is typically named
/// "<program_name>.vollo".
/// The program is intended for a specific hw_config (number of accelerators, cores and other HW
/// configuration options), this function will return an error if any accelerator configuration is
/// incompatible with the program.
/// Once loaded, the program provides inference for several models concurrently.
///
/// The user context given here will be provided on every call to the reader and writer
///
/// Users of this library are free to use the context as they see fit in order to provide a reader
/// and writer to Vollo's configuration bus. The reads and writes will be issued from the same
/// thread as this function call.
///
/// A license needs to be activated with an explicit call to vollo_cfg_activate_license, otherwise
/// an error is returned.
///
/// Once this function returns, as long as there was no error, the Vollo accelerator will be ready
/// to run inferences on. You can load a completely new program by calling this function again. This
/// will also flush any data previously held within the Vollo accelerator.
vollo_cfg_error_t vollo_cfg_load_program(
  void* user_context,
  vollo_cfg_reader_t reader,
  vollo_cfg_writer_t writer,
  const char* program_path);

}  // extern "C"
