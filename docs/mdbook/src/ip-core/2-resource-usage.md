# Resource usage

The following table shows the resource usage of the Vollo IP Core for different configurations. Note,
these resources may vary depending on the version.

The block size determines the side of the matrix block. The core scales with the square of this parameter
e.g. a block size 64 core is around 4 times larger than a block size 32 core.

| Cores | Block size | ALMs | M20Ks | DSPs |
| ----- | ---------- | ---- | ----- | ---- |
| 1     | 32         | 43K  | 1084  | 624  |
| 2     | 32         | 78K  | 2000  | 1248 |
| 3     | 32         | 115K | 2932  | 1872 |
| 4     | 32         | 152K | 3880  | 2496 |
| 5     | 32         | 194K | 4844  | 3120 |
| 6     | 32         | 231K | 5824  | 3744 |
| 1     | 64         | 106K | 3065  | 2400 |
| 2     | 64         | 207K | 5840  | 4800 |
| 3     | 64         | 308K | 8631  | 7200 |
