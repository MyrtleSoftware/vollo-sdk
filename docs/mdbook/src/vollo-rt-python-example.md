# Vollo RT Python Example

The Vollo RT Python bindings are provided for convenience, the runtime
performance of this API is not a priority.

<!-- markdown-link-check-disable -->
Here is a minimal way to use the [Vollo RT Python bindings](./api-reference/vollo_rt.html):
<!-- markdown-link-check-enable -->

```python
import vollo_rt
import torch
import os

with vollo_rt.VolloRTContext() as ctx:
  ctx.add_accelerator(0)

	try:
	  # Try to load the block size 32 identity program
	  ctx.load_program(f"{os.environ["VOLLO_SDK"]}/example/identity_b32.vollo")
	except:
	  # Try to load the block size 64 identity program
	  ctx.load_program(f"{os.environ["VOLLO_SDK"]}/example/identity_b64.vollo")

  input = torch.rand(*ctx.model_input_shape()).bfloat16()
  output = ctx.run(input)

  torch.testing.assert_close(input, output)
  print("Success!")
```
