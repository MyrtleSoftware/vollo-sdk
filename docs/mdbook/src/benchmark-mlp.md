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

### IA-840F: 3 big cores

| Model  | Batch size | Mean latency (μs) | 99th Percentile latency (μs) |
| ------ | ---------- | ----------------- | ---------------------------- |
| mlp_b1 | 1          | 2.6               | 2.8                          |
| mlp_b4 | 4          | 2.9               | 3.1                          |
| mlp_b8 | 8          | 3.6               | 3.8                          |

### IA-420F: 6 small cores

| Model  | Batch size | Mean latency (μs) | 99th Percentile latency (μs) |
| ------ | ---------- | ----------------- | ---------------------------- |
| mlp_b1 | 1          | 3.1               | 3.3                          |
| mlp_b4 | 4          | 3.4               | 3.6                          |
| mlp_b8 | 8          | 4.2               | 4.4                          |
