import torch
import vollo_compiler
import vollo_torch
import numpy as np
import torch.nn as nn


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x)) + x
        return self.out(x)


class ConvBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv = vollo_torch.nn.PaddedConv1d(channels, channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return nn.functional.relu(x)


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


all_configs = {
    "ia_420f": vollo_compiler.Config.ia_420f(),
    "ia_840f": vollo_compiler.Config.ia_840f(),
}

all_models = {
    "identity-128": (Identity(), [128], {}),
    "mlp": (MLP(784, 32, 128), [10, 784], {"unweave": []}),
    "cnn": (CNN(4, 32, 128), [128, 32], {"streaming_transform": [1]}),
}


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        prog=__name__,
        description="Generate and export a Vollo LSTM model to Onnx",
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
    parser.add_argument(
        "-c",
        "--config",
        choices=list(all_configs.keys()),
        help="hardware configuration to use",
    )
    parser.add_argument(
        "-o",
        "--program-out",
        help="Vollo file to export (default: MODEL_NAME.vollo)",
    )
    args = parser.parse_args()

    if args.list_models:
        for model_name in all_configs().keys():
            print(model_name)
        return 0

    if args.model_name is None:
        print("Expected --list-models or --model-name argument")
        parser.print_help()
        return 1

    if args.config is None:
        print("Expected --config argument")
        parser.print_help()
        return 1

    if args.program_out is None:
        args.program_out = f"{args.model_name}.vollo"

    # Create Vollo model
    config = all_configs[args.config]
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
    vm = vollo_compiler.VM.with_program(program)
    actual_y = vm.run(x.detach().numpy())
    np.testing.assert_allclose(
        expected_y.detach().numpy(), actual_y, atol=1e-02, rtol=1e-02
    )

    program_name = args.program_out
    program.save(program_name)
    print(f"Exported model {args.model_name} to {args.program_out}")


if __name__ == "__main__":
    cli()
