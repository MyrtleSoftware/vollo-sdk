# Long Short Term Memory (LSTM) networks

We benchmark an LSTM model consisting of a stack of LSTMs followed by a linear layer.

```python
class LSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super().__init__()
        assert num_layers >= 1
        self.lstm = vollo_torch.nn.LSTM(
            input_size, hidden_size, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.lstm(x)
        x = self.fc(x)
        return x
```

We have also had LSTM models benchmarked and audited as part of a STAC-ML submission where we hold the lowest latency across all models. Please refer to our STAC-ML submissions for more details:

- [STAC ML Sumaco](https://www.stacresearch.com/MRTL221125)

- [STAC ML Tacana](https://www.stacresearch.com/MRTL230426)

### IA-840F: 3 big cores

| Model         | Layers | Hidden size | Parameters | Mean latency (μs) | 99th Percentile latency (μs) |
| ------------- | ------ | ----------- | ---------- | ----------------- | ---------------------------- |
| tiny_lstm     | 2      | 128         | 266K       | 4.3               | 5.1                          |
| small_lstm    | 3      | 256         | 1.6M       | 5.4               | 6.1                          |
| med_lstm      | 3      | 480         | 5.5M       | 8.6               | 9.3                          |
| med_lstm_deep | 6      | 320         | 4.9M       | 8.2               | 8.9                          |

The input size is the same as the hidden size for all models and the output size is set to 32. The layers refers to the number of
LSTM layers in the stack.

### IA-420F: 6 small cores

| Model      | Layers | Hidden size | Parameters | Mean latency (μs) | 99th Percentile latency (μs) |
| ---------- | ------ | ----------- | ---------- | ----------------- | ---------------------------- |
| tiny_lstm  | 2      | 128         | 266K       | 4.9               | 5.7                          |
| small_lstm | 3      | 256         | 1.6M       | 8.9               | 9.6                          |

The input size is the same as the hidden size and the output size is set to 32.

The two medium models are not supported on the IA-420F accelerator card as they are too large to fit in the accelerator memory.
