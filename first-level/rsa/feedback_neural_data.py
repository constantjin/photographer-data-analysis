import gc
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from nipype.interfaces import afni

from ..utils.nifti import load_nifti
from ..utils.path import get_fmriprep_output_dir
from ..utils.subject_exclusion import read_subject_exclusion
from ..utils.types import ConfigDict


def _collect_individual_run_feedback_neural_data(
    subject_id: str, run_id: str, subject_rsa_neural_data_dir: Path, config: ConfigDict
):
    output_dir = Path(config["execution"]["output_dir"])

    run_trial_wise_glm_stat_path = (
        output_dir
        / subject_id
        / run_id
        / "glm_trial_wise"
        / f"{subject_id}_task-photographer_{run_id}_stats.nii"
    )
    try:
        if not (
            subject_rsa_neural_data_dir / run_trial_wise_glm_stat_path.name
        ).exists():
            shutil.copy(run_trial_wise_glm_stat_path, subject_rsa_neural_data_dir)

    except OSError:
        raise RuntimeError(
            f'Cannot copy GLM stat and residual nii file (from <{output_dir / subject_id / run_id / "glm_trial_wise"}> '
            + f"to rsa_neural_data directory <{subject_rsa_neural_data_dir}>."
        )

    os.chdir(subject_rsa_neural_data_dir)

    run_trial_wise_glm_stat_map_name = run_trial_wise_glm_stat_path.name

    # For univariate noise normalization
    if config["execution"]["rsa"]["univariate_noise_normalization"]:
        # Compute standard deviation from residual
        run_glm_residual_path = (
            output_dir
            / subject_id
            / run_id
            / "glm_trial_wise"
            / f"errts.{subject_id}.{run_id}+tlrc"
        )
        run_rsa_residual_stdev_name = (
            f"{subject_id}_task-photographer_{run_id}_glm_residual_stdev.nii"
        )
        try:
            subprocess.run(
                f"3dTstat -prefix {run_rsa_residual_stdev_name} -stdev -overwrite {run_glm_residual_path}",
                shell=True,
            )
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"Cannot compute standard deviation from residual data: <{run_glm_residual_path}>"
            )

        # Apply univariate noise normalization
        run_trial_wise_glm_stat_map_name = (
            f"{subject_id}_task-photographer_{run_id}_norm_stats.nii"
        )
        calc = afni.Calc()
        calc.inputs.in_file_a = f"{subject_id}_task-photographer_{run_id}_stats.nii"
        calc.inputs.in_file_b = run_rsa_residual_stdev_name
        calc.inputs.expr = "a/b"
        calc.inputs.args = "-overwrite"
        calc.inputs.out_file = run_trial_wise_glm_stat_map_name
        calc.run()

    # Select and concatenate trial-wise feedback event beta maps
    trial_list = ["trial3", "trial4", "trial5", "trial6", "trial7", "trial8"]
    rsa_trial_feedback_beta_concat_array = None

    for trial_index in trial_list:
        subbrick_identifier = f"[{trial_index}_feedback#0_Coef]"
        trial_rsa_feedback_beta_map_name = (
            f"{subject_id}_task-photographer_{run_id}_{trial_index}_feedback_beta.nii"
        )

        bucket = afni.Bucket()
        bucket.inputs.in_file = [
            (run_trial_wise_glm_stat_map_name, subbrick_identifier)
        ]
        bucket.inputs.out_file = trial_rsa_feedback_beta_map_name
        bucket.inputs.args = "-overwrite"
        bucket.run()

        trial_rsa_feedback_beta_image = load_nifti(
            subject_rsa_neural_data_dir / trial_rsa_feedback_beta_map_name
        )
        trial_rsa_feedback_beta_array = trial_rsa_feedback_beta_image.data[
            :, :, :, np.newaxis
        ]

        if rsa_trial_feedback_beta_concat_array is None:
            rsa_trial_feedback_beta_concat_array = trial_rsa_feedback_beta_array
        else:
            rsa_trial_feedback_beta_concat_array = np.concatenate(
                (rsa_trial_feedback_beta_concat_array, trial_rsa_feedback_beta_array),
                axis=3,
            )

    # Save the concatenated beta map array into a .npy file
    try:
        subject_rsa_feedback_beta_npy = (
            subject_rsa_neural_data_dir
            / f"{subject_id}_{run_id}_task-photographer_trial_feedback_norm_beta_array.npy"
        )
        np.save(subject_rsa_feedback_beta_npy, rsa_trial_feedback_beta_concat_array)
    except IOError:
        raise RuntimeError(
            f"Cannot save feedback beta array into a .npy file: <{subject_rsa_feedback_beta_npy}>"
        )


def _extract_subject_feedback_neural_data(subject_id: str, config: ConfigDict):
    gc.collect()

    output_dir = Path(config["execution"]["output_dir"])
    assert output_dir.exists(), f"Output directory is not found: <{output_dir}>"

    run_id_list = ["run-01", "run-02", "run-03", "run-04", "run-05"]

    subject_glm_trial_dir_list = [
        output_dir / subject_id / run_id / "glm_trial_wise" for run_id in run_id_list
    ]
    for subject_glm_trial_dir in subject_glm_trial_dir_list:
        if not subject_glm_trial_dir.exists():
            raise RuntimeError(
                'Trial-wise GLM should have done. Please run "glm.run_trial_wise_glm" task first.'
            )

    try:
        subject_rsa_neural_data_dir = (
            output_dir / subject_id / "rsa_neural_data" / "feedback_beta"
        )
        os.makedirs(subject_rsa_neural_data_dir, exist_ok=True)
    except OSError:
        raise RuntimeError(
            f"Cannot create rsa_neural_data directory: <{subject_rsa_neural_data_dir}>"
        )

    for run_id in run_id_list:
        print(subject_id, run_id)

        _collect_individual_run_feedback_neural_data(
            subject_id, run_id, subject_rsa_neural_data_dir, config
        )

    print(f"RSA neural data finished: {subject_id}")


def prepare_feedback_neural_data(config: ConfigDict):
    fmriprep_output_dir = get_fmriprep_output_dir(config)

    # Check subject_exclusion.json present
    try:
        subject_exclusion_dict = read_subject_exclusion(config)
    except RuntimeError as e:
        print(e)
        raise RuntimeError(
            'subject_exclusion.json should be present. Please run both "glm.prepare_task_stim" and "glm.prepare_confound" tasks first.'
        )

    subject_list = [
        child.stem
        for child in fmriprep_output_dir.iterdir()
        if child.is_dir()
        and "sub-" in child.stem
        and child.stem not in subject_exclusion_dict.keys()
    ]

    if config["execution"]["participant_label"] is not None:
        subject_list = [
            child
            for child in subject_list
            if child[4:] in config["execution"]["participant_label"]
        ]

    print(f"Subjects to be processed: {subject_list}")

    for subject_id in subject_list:
        _extract_subject_feedback_neural_data(subject_id, config)
        time.sleep(2)
