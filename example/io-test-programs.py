"""
    CLI to generate and compile Vollo IO only programs
"""

import vollo_compiler


all_configs = {
    "ia_420f_c6b32": vollo_compiler.Config.ia_420f_c6b32(),
    "ia_840f_c3b64": vollo_compiler.Config.ia_840f_c3b64(),
    "ia_840f_c2b64d": vollo_compiler.Config.ia_840f_c2b64d(),
    "v80_c6b32": vollo_compiler.Config.v80_c6b32(),
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog=__name__,
        description="Generate and compile Vollo IO only programs",
    )
    parser.add_argument(
        "input_size",
        type=int,
        help="Size of the input (in bf16 values)",
    )
    parser.add_argument(
        "output_size",
        type=int,
        help="Size of the output (in bf16 values)",
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
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

    args = parser.parse_args()

    if args.config_preset is not None:
        config = all_configs[args.config_preset]
    else:
        config = vollo_compiler.Config.load(args.config)

    input_size = args.input_size
    output_size = args.output_size

    # Program with no compute which aranges IO such that output only starts
    # when all the input is available on the accelerator
    io_only_program = vollo_compiler.Program.io_only_test(
        config, input_size, output_size
    )

    program_name = f"io_only_{input_size}_{output_size}.vollo"
    io_only_program.save(program_name)
    print(program_name)


if __name__ == "__main__":
    main()
