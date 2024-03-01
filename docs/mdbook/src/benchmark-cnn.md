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
| tiny_cnn  | 3      | 128      | 393K       | 4.4               | 5.1                          |
| small_cnn | 3      | 256      | 1.6M       | 5.0               | 5.7                          |
| med_cnn   | 6      | 256      | 3.1M       | 6.3               | 7.0                          |

The kernel size for all models is 8.

### IA-420F: 6 small cores

| Model     | Layers | Channels | Parameters | Mean latency (μs) | 99th Percentile latency (μs) |
| --------- | ------ | -------- | ---------- | ----------------- | ---------------------------- |
| tiny_cnn  | 3      | 128      | 393K       | 5.0               | 5.8                          |
| small_cnn | 3      | 256      | 1.6M       | 4.9               | 5.3                          |
| med_cnn   | 6      | 256      | 3.1M       | 6.1               | 6.9                          |

The kernel size for all models is 8.
