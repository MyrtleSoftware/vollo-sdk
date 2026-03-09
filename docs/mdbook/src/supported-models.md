# Supported Models

The Vollo compiler supports PyTorch models that use the following operations:

| Operation                | Support Notes                                                                                                          | Fp32 Support                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Pointwise arithmetic ops | `+`, `-`, `*`, `/`, `maximum`, `minimum`, the pointwise overload of `max` and `min`                                    | ✅ `+`, `-`, `*`, `maximum`, `minimum` |
| Inequality               | `>`, `<`, `>=`, `<=`                                                                                                   | ✅                                     |
| Clamp ops                | `clamp`, `relu`                                                                                                        | ✅                                     |
| Matrix multiplication    | `Linear`; `matmul` / `@` where one side is a constant                                                                  | ❌                                     |
| Convolution              | Via `vollo_torch.nn.PaddedConv1d`, with `groups == 1` or `groups == in_channels == out_channels`                       | ❌                                     |
| LSTM                     | `torch.nn.LSTM`, `vollo_torch.nn.LSTM`                                                                                 | ❌                                     |
| Indexing / slicing       | Partial square bracket `[]` support; `index_select`, `narrow`                                                          | ✅                                     |
| `sum`                    | `keepdim = True` required when summing over [data dimension](data-dimension.md)                                        | ❌                                     |
| `where`                  | If the `where` condition is an inequality comparison                                                                   | ✅                                     |
| Concatenation            | `cat`, `concat`, `concatenate`                                                                                         | ✅                                     |
| Stacking                 | `stack`, `vstack`, `row_stack`, `hstack`, `column_stack`, `dstack`                                                     | ✅                                     |
| `LayerNorm`              |                                                                                                                        | ❌                                     |
| `RMSNorm`                | via `vollo_torch.nn.RMSNorm` for torch versions < 2.4                                                                  | ❌                                     |
| Batch Normalization      | `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`                                                                            | ✅                                     |
| Transposing              | `transpose`, `swapdims`, `swapaxes`, `t`, `T`, `mT`, `permute`; See [section below](#tensor-memory-format)             | ✅                                     |
| `squeeze`, `unsqueeze`   |                                                                                                                        | ✅                                     |
| Reshaping                | `reshape`, `view`, `reshape_as`, `view_as`, `flatten`; Stride of [data dimension](data-dimension.md) must be unchanged | ✅                                     |
| Broadcasting             | Implicitly or with `broadcast_to`, `broadcast_tensors`, `expand`, `expand_as`                                          | ✅                                     |
| `sqrt`                   | `torch.sqrt`, `torch.rsqrt`                                                                                            | ❌                                     |
| `tanh`                   | `torch.tanh`, `torch.nn.Tanh`                                                                                          | ❌                                     |
| Exponential              | `torch.exp`, `torch.exp2`                                                                                              | ✅                                     |
| `silu`                   | `torch.nn.functional.silu`, `torch.nn.SiLU`                                                                            | ❌                                     |
| `softplus`               | `torch.nn.functional.softplus`, `torch.nn.Softplus`                                                                    | ❌                                     |
| `softmax`                | `torch.softmax`, `torch.nn.Softmax`                                                                                    | ❌                                     |
| `sigmoid`                | `torch.sigmoid`, `torch.nn.functional.sigmoid`, `torch.nn.Sigmoid`                                                     | ❌                                     |

Models that take multiple input tensors and return multiple output tensors
(i.e. a tuple of tensors) are supported.

Note that for operations like `Dropout` and `BatchNorm1d` (which change behaviour at inference time) to be handled correctly, the model should be in `eval` mode.

## Tensor Memory Format

Vollo supports operations on tensors in _data-_ or _channels-_ last memory
format, i.e. the innermost dimension of the tensors should be the _data_ or
_channels_ dimension rather than the _batch_ or _sequence_ dimension if there is
one.
This is because the Vollo accelerator's compute units operate on contiguous
vectors (1D tensors) and has limited support for rearranging tensor data,
particularly transposing them.

There are some notable exceptions that _do not_ require channels-last tensors:

- Layers that operate on sequences: `Conv1d`, `LSTM`.
  Vollo supports the same (_batch_, _channels_, _sequence_) memory format that
  PyTorch uses for these layers, but requires applying the [streaming
  transform](example-2-cnn.md#using-the-streaming-transform) to models that
  contain them.
- General matrix multiplication (as opposed to the more restrictive `Linear`):
  `matmul`, `@`.

## TorchScript

The Vollo compiler supports standard PyTorch modules (`torch.nn.Module`); it
does not support TorchScript modules (`torch.jit.ScriptModule`).
