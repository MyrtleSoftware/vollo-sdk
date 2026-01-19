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

For all the benchmarked models, the input size is the same as the hidden size for all models and the output size is set to 32. The
layers refers to the number of LSTM layers in the stack. The batch size and sequence length are both set to 1 (i.e., we benchmark a
single timestep). Consecutive inferences are run with spacing between them to minimise latency.

We have also had LSTM models benchmarked and audited as part of a STAC-ML submission where we held the lowest latency across all models. Please refer to our STAC-ML submissions for more details:

- [STAC ML Sumaco](https://www.stacresearch.com/MRTL221125)

- [STAC ML Tacana](https://www.stacresearch.com/MRTL230426)

Note that Vollo's current performance, as shown in the tables below, is significantly improved over the STAC-ML submissions.

## V80: 6 core, block size 32

| Model         |   Layers |   Hidden size | Parameters   |   Mean latency (us) |   99th percentile latency (us) |
|---------------|----------|---------------|--------------|---------------------|--------------------------------|
| lstm_tiny     |        2 |           128 | 268K         |                 2.4 |                            2.7 |
| lstm_small    |        3 |           256 | 1.6M         |                 2.8 |                            3.0 |
| lstm_med      |        3 |           480 | 5.6M         |                 3.9 |                            4.2 |
| lstm_med_deep |        6 |           320 | 4.9M         |                 4.2 |                            4.5 |
| lstm_large    |        3 |           960 | 22.2M        |                 8.1 |                            8.4 |

## IA-840F: 3 core, block size 64

| Model         |   Layers |   Hidden size | Parameters   |   Mean latency (us) |   99th percentile latency (us) |
|---------------|----------|---------------|--------------|---------------------|--------------------------------|
| lstm_tiny     |        2 |           128 | 268K         |                 2.2 |                            2.3 |
| lstm_small    |        3 |           256 | 1.6M         |                 2.9 |                            3.0 |
| lstm_med      |        3 |           480 | 5.6M         |                 3.5 |                            3.6 |
| lstm_med_deep |        6 |           320 | 4.9M         |                 3.8 |                            4.0 |

The large model is not supported on the IA-840F accelerator card as it is too large to fit in the accelerator memory.

## IA-420F: 6 core, block size 32

| Model      |   Layers |   Hidden size | Parameters   |   Mean latency (us) |   99th percentile latency (us) |
|------------|----------|---------------|--------------|---------------------|--------------------------------|
| lstm_tiny  |        2 |           128 | 268K         |                 2.2 |                            2.3 |
| lstm_small |        3 |           256 | 1.6M         |                 3.1 |                            3.2 |

The medium and large models are not supported on the IA-420F accelerator card as they are too large to fit in the accelerator memory.
