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

### IA-840F: 3 big cores

| Model     | Layers | Channels | Parameters | Mean latency (μs) | 99th Percentile latency (μs) |
| --------- | ------ | -------- | ---------- | ----------------- | ---------------------------- |
| cnn_tiny  | 3      | 128      | 393K       | 2.9               | 3.3                          |
| cnn_small | 3      | 256      | 1.6M       | 3.1               | 3.5                          |
| cnn_med   | 6      | 256      | 3.1M       | 3.7               | 4.0                          |

The kernel size for all models is 8.

### IA-420F: 6 small cores

| Model     | Layers | Channels | Parameters | Mean latency (μs) | 99th Percentile latency (μs) |
| --------- | ------ | -------- | ---------- | ----------------- | ---------------------------- |
| cnn_tiny  | 3      | 128      | 393K       | 3.0               | 3.4                          |
| cnn_small | 3      | 256      | 1.6M       | 3.4               | 3.9                          |
| cnn_med   | 6      | 256      | 3.1M       | 4.5               | 4.8                          |

The kernel size for all models is 8.
