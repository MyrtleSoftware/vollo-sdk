# Supported Models

The VOLLO compiler supports PyTorch models that use the following operations:

| Operation                | Support Notes                                                  |
| ------------------------ | -------------------------------------------------------------- |
| Pointwise arithmetic ops | `+`, `-`, `*`; `/` by constant                                 |
| Inequality               | `>`, `<`, `>=`, `<=`                                           |
| `max` and `min`          |                                                                |
| Clamp ops                | `clamp`, `relu`                                                |
| Matrix multiplication    | `Linear`; `matmul` / `@` where one side is a constant          |
| Convolution              | Via `vollo_torch.nn.PaddedConv1d`                              |
| Indexing / slicing       | Partial square bracket `[]` support; `index_select`            |
| `sum`                    | With `keepdim=True`                                            |
| `where`                  | If the `where` condition is an inequality comparison           |
| Concatenation            | `cat`, `concat` on outer dimension or at start or end of model |
| `transpose`              | See [section below](#tensor-memory-format)                     |

## Tensor Memory Format

VOLLO supports operations on tensors in channels-last memory format, and does
not currently support changing the memory format on the accelerator (e.g.
transpose operations).

The compiler *is* able to support many common cases of models where the PyTorch
memory format is not channels-last, or that include transposes.
In particular, [PyTorch's 1D convolutions](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
have sequence length last rather than channels last but the compiler can
[transform](example-3-cnn.md#using-the-streaming-transform) them to use
channels last.

## TorchScript

The VOLLO compiler supports standard PyTorch modules (`torch.nn.Module`); it
does not support TorchScript modules (`torch.jit.ScriptModule`).
