"""
    CLI to generate and compile Vollo reference models.
"""

import argparse
import json
import warnings

import torch

import vollo_compiler
import vollo_torch
import numpy as np
import torch.nn as nn

# In some versions of torch, a module within torch (torch.fx.proxy) triggers a
# deprecation warning from another module of torch (torch.overrides), so just
# ignore these.
warnings.filterwarnings("ignore", category=UserWarning, module="torch.overrides")


def pretty_parameters(num_params: int) -> str:
    """
    >>> pretty_parameters(12345)
    "12K"
    >>> pretty_parameters(123456)
    "123K"
    >>> pretty_parameters(1234567)
    "1.2M"
    >>> pretty_parameters(12345678)
    "12.3M"
    """
    if num_params < 1000:
        return str(num_params)
    elif num_params < 1_000_000:
        # K range
        k_value = num_params / 1000
        if k_value >= 100:
            return f"{int(k_value)}K"
        elif k_value >= 10:
            return f"{k_value:.0f}K"
        else:
            return f"{k_value:.1f}K"
    else:
        # M range
        m_value = num_params / 1_000_000
        if m_value >= 100:
            return f"{int(m_value)}M"
        elif m_value >= 10:
            return f"{m_value:.1f}M"
        else:
            return f"{m_value:.1f}M"


class Identity(torch.nn.Module):
    def forward(self, x):
        return x

    def describe_self(self, _input_shape):
        return {
            "Model": "Identity",
        }


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

    def describe_self(self, input_shape):
        batch_size = input_shape[0]
        return {
            "Model": f"mlp_b{batch_size}",
            "Batch size": batch_size,
        }


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

    def describe_self(self, input_shape):
        return {
            "Layers": len(self.cnn),
            "Channels": input_shape[1],
            "Parameters": pretty_parameters(sum(p.numel() for p in self.parameters())),
        }


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

    def describe_self(self, input_shape):
        return {
            "Layers": self.lstm.lstm.num_layers,
            "Hidden size": self.lstm.lstm.hidden_size,
            "Parameters": pretty_parameters(sum(p.numel() for p in self.parameters())),
        }


all_configs = {
    "ia_420f_c6b32": vollo_compiler.Config.ia_420f_c6b32(),
    "ia_840f_c3b64": vollo_compiler.Config.ia_840f_c3b64(),
    "ia_840f_c2b64d": vollo_compiler.Config.ia_840f_c2b64d(),
    "v80_c6b32": vollo_compiler.Config.v80_c6b32(),
}

# The number of timesteps in the streaming dimension of the model
# NOTE: as the models are using a streaming transformation, this dimension will be streamed over:
# the timesteps are computed sequentially, and we are interested in the latency per timestep.
# the value of STREAM_DIM is not important.
STREAM_DIM = 1

all_models = {
    "identity-128": (Identity(), [128], {}),
    # mlp models
    "mlp_b1": (MLP(256, 384, 128), [1, 256], {}),
    "mlp_b4": (MLP(256, 384, 128), [4, 256], {}),
    "mlp_b8": (MLP(256, 384, 128), [8, 256], {}),
    # cnn models
    "cnn_tiny": (CNN(3, 8, 128), [128, STREAM_DIM], {"streaming_transform": [1]}),
    "cnn_small": (CNN(3, 8, 256), [256, STREAM_DIM], {"streaming_transform": [1]}),
    "cnn_med": (CNN(6, 8, 256), [256, STREAM_DIM], {"streaming_transform": [1]}),
    # lstm models
    "lstm_tiny": (
        LSTM(2, 128, 128, 32),
        [STREAM_DIM, 128],
        {"streaming_transform": [0]},
    ),
    "lstm_small": (
        LSTM(3, 256, 256, 32),
        [STREAM_DIM, 256],
        {"streaming_transform": [0]},
    ),
    "lstm_med": (
        LSTM(3, 480, 480, 32),
        [STREAM_DIM, 480],
        {"streaming_transform": [0]},
    ),
    "lstm_med_deep": (
        LSTM(6, 320, 320, 32),
        [STREAM_DIM, 320],
        {"streaming_transform": [0]},
    ),
}


def get_selected_subparser(parser, args):
    """
    Get the parser that was selected based on the subcommand
    """
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        choice = getattr(args, action.dest)
        if choice is None:
            return None
        else:
            subparser = action.choices[choice]
            return get_selected_subparser(subparser, args) or subparser
    else:
        return None


def list_models(parser, args):
    for model_name in all_models.keys():
        if args.model_type == "all" or model_name.startswith(args.model_type):
            print(model_name)
    return 0


def compile_model(parser, args):
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

    if args.describe_only:
        description = model.describe_self(input_shape)
        print(json.dumps(description, indent=2))
        return 0

    x = torch.randn(input_shape).bfloat16().to(torch.float32)

    model.eval()
    model, expected_y = vollo_torch.fx.prepare_shape(model, x)

    nnir_graph = vollo_torch.fx.nnir.to_nnir(model)

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
    vm = program.to_vm(bit_accurate=False)
    actual_y = vm.run(x.detach().numpy())
    np.testing.assert_allclose(
        expected_y.detach().numpy(), actual_y, atol=1e-02, rtol=1e-02
    )
    print(f"VM output matches expected output")
    print(f"Took {vm.cycle_count()} cycles")

    program_name = args.program_out
    program.save(program_name)
    print(f"Exported model {args.model_name} to {args.program_out}")


def cli():
    parser = argparse.ArgumentParser(
        prog="programs.py",
        description="Generate and compile a Vollo reference model",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_models_parser = subparsers.add_parser("list-models")
    list_models_parser.add_argument(
        "--model-type", choices=["all", "identity", "mlp", "cnn", "lstm"], default="all"
    )
    list_models_parser.set_defaults(handler=list_models)

    compile_model_parser = subparsers.add_parser("compile-model")
    compile_model_parser.set_defaults(handler=compile_model)
    compile_model_parser.add_argument(
        "-m",
        "--model-name",
        choices=list(all_models.keys()),
        help="name of the Vollo model to generate",
    )
    config_group = compile_model_parser.add_mutually_exclusive_group()
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
    compile_model_parser.add_argument(
        "-o",
        "--program-out",
        help="Vollo file to export (default: MODEL_NAME.vollo)",
    )
    compile_model_parser.add_argument(
        "--describe-only",
        action="store_true",
        help="output json describing the model and exit (requires --model-name)",
    )

    args = parser.parse_args()

    return args.handler(get_selected_subparser(parser, args), args)


if __name__ == "__main__":
    cli()
