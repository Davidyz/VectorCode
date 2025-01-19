import logging
import os

from vectorcode import __version__
from vectorcode.cli_utils import (
    CliAction,
    cli_arg_parser,
    find_project_config_dir,
    load_config_file,
)
from vectorcode.subcommands import drop, init, ls, query, vectorise


def main():
    cli_args = cli_arg_parser()
    project_config_dir = find_project_config_dir(cli_args.project_root)

    if project_config_dir is not None:
        project_config_file = os.path.join(project_config_dir, "config.json")
        if os.path.isfile(project_config_file):
            final_configs = load_config_file(project_config_file).merge_from(cli_args)
        else:
            final_configs = cli_args
    else:
        final_configs = load_config_file().merge_from(cli_args)

    if final_configs.pipe:
        # NOTE: NNCF (intel GPU acceleration for sentence transformer) keeps showing logs.
        # This disables logs below ERROR so that it doesn't hurt the `pipe` output.
        logging.disable(logging.ERROR)

    return_val = 0
    match final_configs.action:
        case CliAction.query:
            return_val = query(final_configs)
        case CliAction.vectorise:
            return_val = vectorise(final_configs)
        case CliAction.drop:
            return_val = drop(final_configs)
        case CliAction.ls:
            return_val = ls(final_configs)
        case CliAction.init:
            return_val = init(final_configs)
        case CliAction.version:
            print(__version__)
            return_val = 0
    return return_val
