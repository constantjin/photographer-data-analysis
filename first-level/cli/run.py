import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import toml

from ..utils.types import ConfigDict

DEFAULT_CONFIG_FILE_NAME = "photographer_config.toml"


def main():
    def _path_exists(path, parser: ArgumentParser):
        if path is None or not Path(path).exists():
            raise parser.error(f"Path does not exist: <{path}>")

        return Path(path).absolute()

    def _path_abs(path):
        return Path(path).absolute()

    def _drop_sub(sub_input: str):
        return sub_input[4:] if sub_input.startswith("sub-") else sub_input

    parser = ArgumentParser(description="Photographer Data First-level Analysis")

    PathExists = partial(_path_exists, parser=parser)

    # Required arguments
    parser.add_argument(
        "bids_dir", action="store", type=PathExists, help="The BIDS root directory."
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=_path_abs,
        help="The analysis output directory. It should be (bids_dir)/derivatives/first-level",
    )
    parser.add_argument(
        "analysis_level",
        choices=["participant"],
        help='Processing stage. Only "participant" is accepted. I believe this is in the BIDS-Apps spec?',
    )

    g_bids = parser.add_argument_group("BIDS-related argument")
    g_bids.add_argument(
        "--participant-label",
        "--participant_label",
        action="store",
        nargs="+",
        type=_drop_sub,
        help="A single participant label or a space-separated participant labels.",
    )

    g_step = parser.add_argument_group("First-level analysis-related arguments")
    g_step.add_argument(
        "-t",
        "--task",
        choices=[
            # For GLM-related tasks
            "glm.prepare_task_stim",
            "glm.prepare_confound",
            "glm.run_block_wise_glm",
            "glm.run_trial_wise_glm",
            # For the MNI-based gray matter mask
            "mask.prepare_gm_mask",
            # For behavioral analyses and feedback model RDMs
            "behavior.prepare_behavioral_data",
            # For RSA analyses
            "rsa.prepare_feedback_neural_data",
            "rsa.prepare_feedback_model_rdm",
            "rsa.run_feedback_rsa",
            # For statistical analyses
            "stat.run_univariate_ttest",
            "stat.run_feedback_rsa_ttest",
            "stat.extract_feedback_rsa_cluster_mask",
        ],
        action="store",
        required=True,
        help="A first-level analysis task to run.",
    )
    g_step.add_argument(
        "--config-file",
        "--config_file",
        type=_path_abs,
        action="store",
        help=f"A config file (toml) path. If not specified, we will try to find {DEFAULT_CONFIG_FILE_NAME} in (bids_dir)/code.",
    )

    arg_opt = parser.parse_args()

    # Validate arguments
    config_file_path = (
        arg_opt.config_file
        if arg_opt.config_file is not None
        else arg_opt.bids_dir / "code" / DEFAULT_CONFIG_FILE_NAME
    )
    if not Path(config_file_path).exists():
        parser.error(f"Config file does not exist: <{config_file_path}>")

    if arg_opt.output_dir != arg_opt.bids_dir / "derivatives" / "first-level":
        parser.error("Output directory should be (bids_dir)/derivatives/first-level.")

    # Read the config toml file
    with open(config_file_path, "r") as f:
        config_toml_data = toml.load(f)

    # Put arguments/config data into one dictionary
    config = {}
    config["execution"] = vars(arg_opt) | config_toml_data
    config["execution"]["subject_exclusion_file_path"] = (
        config["execution"]["output_dir"] / "subject_exclusion.json"
    )
    config = ConfigDict(config)

    # Disable nipype etelemetry
    os.environ["NIPYPE_NO_ET"] = "True"
    os.environ["NO_ET"] = "True"

    # Create output directory
    os.makedirs(config["execution"]["output_dir"], exist_ok=True)

    # Run specific analysis step
    from ..behavior.behavioral_data import prepare_behavioral_data
    from ..glm.confound import prepare_confound
    from ..glm.glm_block_wise import run_block_wise_glm
    from ..glm.glm_trial_wise import run_trial_wise_glm
    from ..glm.task_stim import prepare_task_stim
    from ..mask.gm_mask import prepare_gm_mask
    from ..rsa.feedback_model_rdm import prepare_feedback_model_rdm
    from ..rsa.feedback_neural_data import prepare_feedback_neural_data
    from ..rsa.feedback_rsa import run_feedback_rsa
    from ..stat.feedback_rsa_cluster_mask import extract_feedback_rsa_cluster_mask
    from ..stat.feedback_rsa_ttest import run_feedback_rsa_ttest
    from ..stat.univariate_ttest import run_univariate_ttest

    task = config["execution"]["task"]

    # For GLM-related tasks
    if task == "glm.prepare_task_stim":
        prepare_task_stim(config)

    elif task == "glm.prepare_confound":
        prepare_confound(config)

    elif task == "glm.run_block_wise_glm":
        run_block_wise_glm(config)

    elif task == "glm.run_trial_wise_glm":
        run_trial_wise_glm(config)

    # For the MNI-based gray matter mask
    elif task == "mask.prepare_gm_mask":
        prepare_gm_mask(config)

    # For behavioral analyses and feedback model RDMs
    elif task == "behavior.prepare_behavioral_data":
        prepare_behavioral_data(config)

    # For RSA analyses
    elif task == "rsa.prepare_feedback_neural_data":
        prepare_feedback_neural_data(config)

    elif task == "rsa.prepare_feedback_model_rdm":
        prepare_feedback_model_rdm(config)

    elif task == "rsa.run_feedback_rsa":
        run_feedback_rsa(config)

    # For statistical analyses
    elif task == "stat.run_univariate_ttest":
        run_univariate_ttest(config)

    elif task == "stat.run_feedback_rsa_ttest":
        run_feedback_rsa_ttest(config)

    elif task == "stat.extract_feedback_rsa_cluster_mask":
        extract_feedback_rsa_cluster_mask(config)

    else:
        parser.error(f"Cannot find modules for the input task name: {task}")
