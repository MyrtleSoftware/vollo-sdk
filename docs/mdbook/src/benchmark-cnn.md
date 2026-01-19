# 1D Convolutional neural networks (CNN)

We benchmark a simple 1-D convolutional model with a residual connection after every layer.

```python
class ConvBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv = vollo_torch.nn.PaddedConv1d(channels, channels, kernel_size)

    def forward(self, inp):
        x = self.conv(inp)
        return nn.functional.relu(x) + inp


class CNN(nn.Module):
    def __init__(self, num_layers, kernel_size, channels):
        super().__init__()
        assert num_layers >= 1

        self.cnn = nn.Sequential(
            *[ConvBlock(channels, kernel_size) for i in range(num_layers)],
        )

    def forward(self, x):
        x = self.cnn(x)  # N x channels x T
        return x
```

The kernel size for all models is 8. The batch size and sequence length are both set to 1 (i.e., we benchmark a single timestep).
Consecutive inferences are run with spacing between them to minimise latency.

## V80: 6 cores, block size 32

| Model     |   Layers |   Channels | Parameters   |   Mean latency (us) |   99th percentile latency (us) |
|-----------|----------|------------|--------------|---------------------|--------------------------------|
| cnn_tiny  |        3 |        128 | 393K         |                 2.5 |                            2.8 |
| cnn_small |        3 |        256 | 1.6M         |                 2.4 |                            2.7 |
| cnn_med   |        6 |        256 | 3.1M         |                 3.3 |                            3.4 |

## IA-840F: 3 cores, block size 64

| Model     |   Layers |   Channels | Parameters   |   Mean latency (us) |   99th percentile latency (us) |
|-----------|----------|------------|--------------|---------------------|--------------------------------|
| cnn_tiny  |        3 |        128 | 393K         |                 2.1 |                            2.3 |
| cnn_small |        3 |        256 | 1.6M         |                 2.5 |                            2.7 |
| cnn_med   |        6 |        256 | 3.1M         |                 3.1 |                            3.3 |

## IA-420F: 6 cores, block size 32

| Model     |   Layers |   Channels | Parameters   |   Mean latency (us) |   99th percentile latency (us) |
|-----------|----------|------------|--------------|---------------------|--------------------------------|
| cnn_tiny  |        3 |        128 | 393K         |                 2.2 |                            2.3 |
| cnn_small |        3 |        256 | 1.6M         |                 2.7 |                            2.9 |
| cnn_med   |        6 |        256 | 3.1M         |                 3.4 |                            3.6 |
