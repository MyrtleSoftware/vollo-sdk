# Multilayer perceptron (MLP)

The model below is a simple multilayer perceptron (MLP) with 3 layers.

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)

mlp = MLP(256.0, 384.0, 128.0)
```

We demonstrate the model at a variety of batch sizes. The model has 295K parameters.

## V80: 6 cores, block size 32

> <div class="warning">
> V80 PCIe optimisations underway, improvements coming in the next release
> </div>

| Model   |   Batch size |   Mean latency (us) |   99th percentile latency (us) |
|---------|--------------|---------------------|--------------------------------|
| mlp_b1  |            1 |                 2.8 |                            3.0 |
| mlp_b4  |            4 |                 3.8 |                            4.0 |
| mlp_b8  |            8 |                 5.9 |                            6.2 |

## IA-840F: 3 cores, block size 64

| Model  | Batch size | Mean latency (μs) | 99th Percentile latency (μs) |
| ------ | ---------- | ----------------- | ---------------------------- |
| mlp_b1 | 1          | 2.3               | 2.4                          |
| mlp_b4 | 4          | 2.5               | 2.6                          |
| mlp_b8 | 8          | 2.7               | 2.8                          |

## IA-420F: 6 core, block size 32

| Model  | Batch size | Mean latency (μs) | 99th Percentile latency (μs) |
| ------ | ---------- | ----------------- | ---------------------------- |
| mlp_b1 | 1          | 2.9               | 3.0                          |
| mlp_b4 | 4          | 3.0               | 3.1                          |
| mlp_b8 | 8          | 3.4               | 3.5                          |
