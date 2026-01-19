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

| Model   |   Batch size |   Mean latency (us) |   99th percentile latency (us) |
|---------|--------------|---------------------|--------------------------------|
| mlp_b1  |            1 |                 2.3 |                            2.5 |
| mlp_b4  |            4 |                 2.8 |                            3.1 |
| mlp_b8  |            8 |                 3.4 |                            4.0 |

## IA-840F: 3 cores, block size 64

| Model   |   Batch size |   Mean latency (us) |   99th percentile latency (us) |
|---------|--------------|---------------------|--------------------------------|
| mlp_b1  |            1 |                 2.5 |                            2.6 |
| mlp_b4  |            4 |                 2.7 |                            3.0 |
| mlp_b8  |            8 |                 3.2 |                            3.5 |

## IA-420F: 6 core, block size 32

| Model   |   Batch size |   Mean latency (us) |   99th percentile latency (us) |
|---------|--------------|---------------------|--------------------------------|
| mlp_b1  |            1 |                 2.5 |                            2.7 |
| mlp_b4  |            4 |                 3.1 |                            3.3 |
| mlp_b8  |            8 |                 4.1 |                            4.4 |
