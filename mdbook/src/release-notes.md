# Release Notes

## 0.10.1

* Support for scalar (`int`, `float`) literals in pointwise operations in
  `vollo-torch`.

## 0.10.0

* Architectural changes in bitstream to support compiler
* Reduced latency from reduced core to core communication in the bitstream
* Add general model compiler and VM simulation with Python bindings in
  `vollo-python`
* Add PyTorch frontend to model compiler in `vollo-torch`

## 0.5.4

* Fix performance bug on the IA420F device

## 0.5.3

* Support for programming the device flash over PCIe in vollo-tool
* Support for reading device temperature in vollo-tool

## 0.5.2

* Better support for multi-layer perceptron models
* Performance improvements for IO bound models
* Bugfix in pointwise unit

## 0.5.1

* Support for IA420F device added. Vollo compiler now requires a bitstream info
  argument e.g. `vollo-compiler --bitstream bitstream/vollo-ia840f.json ...`

## 0.5.0

* Performance optimization for models which do not use the LSTM
* Add `bitstream/vollo-ia840f.json`, which contains the configuration
  information for the provided bitstream
* Add `bitstream-info` subcommand to `vollo-tool`, which checks which bitstream
  is configured on any Vollo devices
* Add `bitstream-check` subcommand to `vollo-tool`, which checks the bitstream
  configured on any Vollo devices against a given configuration
* Updated licensing and DRM

## 0.4.1

* No longer requires root access to interface with the vollo
* No longer require hugepages
* No longer require the iommu to be deactivated. The iommu should still be
  deactivated on workloads across multiple boards
* Fixes `vollo_add_job_fp32`
