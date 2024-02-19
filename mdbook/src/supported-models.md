# Supported Models

The VOLLO compiler supports PyTorch models that use the following operations:

| Operation                  | Support Notes                                         |
|----------------------------|-------------------------------------------------------|
| Pointwise arithmetic ops   | `+`, `-`, `*`; `/` by constant                        |
| Inequality                 | `>`, `<`, `>=`, `<=`                                  |
| `max` and `min`            |                                                       |
| Clamp ops                  | `clamp`, `relu`                                       |
| Matrix multiplication      | `Linear`; `matmul` / `@` where one side is a constant |
| Convolution                | Via `vollo_torch.nn.PaddedConv1d`                     |
| Indexing / slicing         | Partial square bracket `[]` support; `index_select`   |
| `sum`                      | With `keepdim=True`                                   |
| `where`                    | If the `where` condition is an inequality comparison  |
| Concatenation              | `cat`, `concat` on outer dimension if data is aligned |
| `transpose`                | See footnote[^transpose]                              |

[^transpose]: The compiler will The VOLLO accelerator does not currently support
    transposes at compute time, but the compiler will remove transposes from the
    model if it is able to.
