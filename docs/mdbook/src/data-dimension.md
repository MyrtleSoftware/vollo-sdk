# The data dimension

For a model to compile on Vollo, in addition to the normal tensor rank/extent
constraints on algorithms/functions, an additional constraint on the _data
dimension_ must be satisfied. Conceptually, the data dimension is the
contiguous dimension of a tensor, it transforms and constrains algorithms
according to the rules below.

In most use cases the data dimension is completely opaque, the Vollo
compiler will deduce it and the program will compile without changes.

For the remainder of this page:

- We represent an `n`-dimensional (a.k.a. rank-`n`) tensors as `[a b! c]`, in
  this example a rank-3 tensor with extents `a`, `b` and, `c`.
- The dimension with a `!` is the data dimension.
- Tensors that are compile-time constants (like weights) don't have a data
  dimension.

## Pointwise

Shape **and** data dimension must match.

```txt
[a b! c] (*) [a b! c] -> [a b! c]
```

Where `(*)` is any pointwise operation, e.g. `+`, `-`, `*`, `/`, `maximum`,
`minimum`, the pointwise overload of `max` and `min`, etc.

## Slicing

This preserves the data dimension.

A slice on the data dimension:

```txt
[a b c!][:, :, :n] -> [a b n!]
```

Non data-dimension slice:

```txt
[a b! c][:, :, :n] -> [a b! n]
```

A non data-dimension slice is free (no compute).

## Unsqueeze (a.k.a. new-axis)

You can add a new axis anywhere, the new axis is never the data dimension:

```txt
[a! b].unsqueeze(dim=0) -> [1 a! b]
```

## Broadcasting

You can broadcast along a non data-dimension:

```txt
[1 b!] -> [n b!]
```

Or along the data dimension:

```txt
[a 1!] -> [a n!]
```

Broadcasting a non data-dimension is free (no compute), and broadcasting the
data dimension is close to free.

## Concatenation

Similar to a pointwise operation, shape **and** data dimension of each
concatenated tensor must match.

Along the data dimension:

```txt
[a! b c].repeat(n, dim=0) -> [(a * n)! b c]
```

Non data-dimension:

```txt
[a! b c].repeat(n, dim=1) -> [a! (b * n) c]
```

A non data-dimension concatenation is free (no compute).

Note: stacking is the same but with a new-axis before the concatenation.

## Reduction

In general reductions preserve the position of the data dimension.

Along the data dimension:

```txt
[a! b].sum(dim=0) -> [1! b]
```

Non data-dimension reduction (generally slower):

```txt
[a! b].sum(dim=1) -> [a!]
```

Note: `keepdim` must be used in the former and is optional in the latter.

## Matrix multiplication

These operations transform the data dimension in non-obvious ways, here we use
`*` do denote any number of commensurate broadcast dimensions, none of which
are allowed to be the data dimension.

With one side a compile-time constant, in this case the LHS (WLOG):

```txt
[* i j] @ [* j! k] -> [* i! k]
```

That is, the data dimension must be along the contracted dimension of the
runtime tensor. The output data dimension is along the "replaced" index.

Note: a linear layer is a special case of the above with the `k` dimension
squeezed out.

If the contraction is not along the data dimension of the non-constant
(requires `allow_dynamic_weights`):

```txt
[* i j] @ [* j k!] -> [* i k!]
```

That is, the data dimension of non-constant is preserved in the output. This
potentially has a higher latency and consumes more tensor RAM than contracting
along the data dimension.

With both sides runtime tensors (also requires `allow_dynamic_weights`):

```txt
[* i! j] @ [* j! k] -> [* i! k]
```

That is, the contracted dimension must be the data dimension of exactly one of
the input tensors (in this case the RHS WLOG). The output data dimension is
that of the side whose data dimension was not contracted, for example:

```txt
[* i j!] @ [* j k!] -> [* i k!]
```

This uses more tensor RAM and potentially has a higher latency than
contractions with a constant.

## Transpose

Tensors can be transposed without restriction. If one of the transposed
dimensions is the data dimension, the data dimension is transposed to that
dimension:

```txt
[a! b c d].transpose(0, 2) -> [c b a! d]
```

## Reshape

For a given tensor's extents, e.g. `[a b c]`, each dimension has a _stride_ equal
to the product of all dimensions to its right, i.e. `[(b * c) c 1]`. The stride
of the data dimension must/will be preserved during a reshape. For example:

```txt
[a! (b * c)] -> [a! b c]
```

Is valid because the stride of the data dimension is `b * c` before and after.

```txt
[a b c!] -> [b a c!]
```

Is valid because the stride of the data dimension is `1` before and after.

```txt
[a b! c] -> [(a * b)! c]
```

Is valid because the stride of the data dimension is `c` before and after, similarly:

```txt
[a (n * b)! c] -> [a n b! c]
```

Note: that the resultant data dimension is deduced to `b!` rather than `n!` to
uphold the stride requirement.

However:

```txt
[a b! c] -> [a (b * c)!]
```

Is invalid because the stride of the data dimension is `c` before but `1` after.

If the output shape has multiple candidate dimensions with the input data dimension's
stride (note that these candidate dimensions are all consecutive and all but the
leftmost have extent 1), the leftmost of them will be chosen as the output data
dimension:

```text
[a! b] -> [a! 1 b]
```

A reshape that doesn't change the extent of the data dimension is free (no compute).

Note: the strides discussed in this subsection are conceptual and not related
to the strides of the tensors queryable from PyTorch etc.
