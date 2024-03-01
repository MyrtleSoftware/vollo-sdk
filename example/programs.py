"""
    CLI to generate and compile VOLLO reference models.
"""

import torch

import vollo_compiler
import vollo_torch
import numpy as np
import torch.nn as nn


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.out(x)


# A simple convolutional block with a residual connection
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


class LSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super().__init__()
        assert num_layers >= 1
        self.lstm = vollo_torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.lstm(x)
        x = self.fc(x)
        return x


all_configs = {
    "ia_420f": vollo_compiler.Config.ia_420f_c6b32(),
    "ia_840f": vollo_compiler.Config.ia_840f_c3b64(),
}

# The number of timesteps in the streaming dimension of the model
# NOTE: as the models are using a streaming transformation, this dimension will be streamed over:
# the timesteps are computed sequentially, and we are interested in the latency per timestep.
# the value of STREAM_DIM is not important.
STREAM_DIM = 32

all_models = {
    "identity-128": (Identity(), [128], {}),
    # mlp models
    "mlp_b1": (MLP(256, 384, 128), [1, 256], {"unweave": []}),
    "mlp_b4": (MLP(256, 384, 128), [4, 256], {"unweave": []}),
    "mlp_b8": (MLP(256, 384, 128), [8, 256], {"unweave": []}),
    # cnn models
    "tiny_cnn": (CNN(3, 8, 128), [128, STREAM_DIM], {"streaming_transform": [1]}),
    "small_cnn": (CNN(3, 8, 256), [256, STREAM_DIM], {"streaming_transform": [1]}),
    "med_cnn": (CNN(6, 8, 256), [256, STREAM_DIM], {"streaming_transform": [1]}),
    # lstm models
    "tiny_lstm": (
        LSTM(2, 128, 128, 32),
        [STREAM_DIM, 128],
        {"streaming_transform": [0]},
    ),
    "small_lstm": (
        LSTM(3, 256, 256, 32),
        [STREAM_DIM, 256],
        {"streaming_transform": [0]},
    ),
    "med_lstm": (
        LSTM(3, 480, 480, 32),
        [STREAM_DIM, 480],
        {"streaming_transform": [0]},
    ),
    "med_lstm_deep": (
        LSTM(6, 320, 320, 32),
        [STREAM_DIM, 320],
        {"streaming_transform": [0]},
    ),
}


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        prog=__name__,
        description="Generate and compile a VOLLO reference model",
    )
    parser.add_argument(
        "-l",
        "--list-models",
        action="store_true",
        help="list the names of all supported models and exit",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        choices=list(all_models.keys()),
        help="name of the Vollo model to generate",
    )
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "-c",
        "--config",
        help="hardware configuration JSON to use",
    )
    config_group.add_argument(
        "--config-preset",
        choices=list(all_configs.keys()),
        help="hardware configuration preset to use",
    )
    parser.add_argument(
        "-o",
        "--program-out",
        help="Vollo file to export (default: MODEL_NAME.vollo)",
    )
    args = parser.parse_args()

    if args.list_models:
        for model_name in all_models.keys():
            print(model_name)
        return 0

    if args.model_name is None:
        parser.error("Expected --list-models or --model-name argument")

    if args.config is None and args.config_preset is None:
        parser.error("Expected --config or --config-preset argument")

    if args.program_out is None:
        args.program_out = f"{args.model_name}.vollo"

    # Create Vollo model
    if args.config_preset is not None:
        config = all_configs[args.config_preset]
    else:
        config = vollo_compiler.Config.load(args.config)

    model, input_shape, transforms = all_models[args.model_name]

    x = torch.randn(input_shape).bfloat16().to(torch.float32)

    model, expected_y = vollo_torch.fx.prepare_shape(model, x)

    nnir_graph = vollo_torch.fx.nnir.to_nnir(model)

    if "unweave" in transforms:
        nnir_graph = nnir_graph.unweave()
    if "streaming_transform" in transforms:
        input_streaming_axis = transforms["streaming_transform"][0]
        (nnir_graph, output_streaming_axis) = nnir_graph.streaming_transform(
            input_streaming_axis
        )
        expected_y = torch.index_select(
            expected_y, output_streaming_axis, torch.tensor([0])
        )
        expected_y = torch.squeeze(expected_y, output_streaming_axis)
        x = torch.index_select(x, input_streaming_axis, torch.tensor([0]))
        x = torch.squeeze(x, input_streaming_axis)

    program = nnir_graph.to_program(config)

    # run in the VM
    vm = program.to_vm()
    actual_y = vm.run(x.detach().numpy())
    np.testing.assert_allclose(
        expected_y.detach().numpy(), actual_y, atol=1e-02, rtol=1e-02
    )
    print(f"VM output matches expected output")
    print(f"Took {vm.cycle_count()} cycles")

    program_name = args.program_out
    program.save(program_name)
    print(f"Exported model {args.model_name} to {args.program_out}")


if __name__ == "__main__":
    cli()
