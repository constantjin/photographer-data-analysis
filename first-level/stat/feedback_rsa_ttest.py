import gc
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..utils.nifti import load_nifti
from ..utils.path import get_fmriprep_output_dir
from ..utils.subject_exclusion import read_subject_exclusion
from ..utils.types import ConfigDict


def run_feedback_rsa_ttest(config: ConfigDict):
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

    output_dir = Path(config["execution"]["output_dir"])
    assert output_dir.exists(), f"Output directory is not found: <{output_dir}>"

    rsa_feedback_model_name_list = [
        "current_trial",
        "one_back_trial",
        "two_back_trial",
        "recent_2_trial",
        "recent_3_trial",
        "previous_2_trial",
    ]

    searchlight_radius = config["execution"]["rsa"]["searchlight_radius"]
    blur_kernel_width = config["execution"]["rsa"]["rsa_blur_kernel_width"]

    run_id_list = ["run-01", "run-02", "run-03", "run-04", "run-05"]

    for rsa_feedback_model_name in rsa_feedback_model_name_list:
        gc.collect()

        print(f"T-test on {rsa_feedback_model_name} RSA maps (average across all runs)")

        rsa_map_name = f"{rsa_feedback_model_name}"

        # Create stat - ttest directory
        try:
            stat_ttest_dir = (
                output_dir
                / "stat"
                / "multivariate"
                / "feedback_model"
                / "ttest"
                / rsa_map_name
            )
            os.makedirs(stat_ttest_dir, exist_ok=True)
        except OSError:
            raise RuntimeError(
                f"Cannot create t-test output directory: <{stat_ttest_dir}>"
            )

        # Copy GM mask and MNI template
        try:
            mni_gm_mask_path = output_dir / "mask" / "mni_152_gm_mask_3mm.nii"
            if not (stat_ttest_dir / mni_gm_mask_path.name).exists():
                shutil.copy(mni_gm_mask_path, stat_ttest_dir)
        except OSError:
            raise RuntimeError(
                f"Cannot copy MNI152 GM mask (from <{mni_gm_mask_path}>) to t-test output directory (<{stat_ttest_dir}>)."
            )

        try:
            afni_template_path = (
                Path(config["execution"]["glm"]["afni_path"])
                / "MNI152_2009_template_SSW.nii.gz"
            )
            if not (stat_ttest_dir / afni_template_path.name).exists():
                shutil.copy(afni_template_path, stat_ttest_dir)
        except OSError:
            raise RuntimeError(
                f"Cannot copy AFNI template (from <{afni_template_path}>) to t-test output directory (<{stat_ttest_dir}>)."
            )

        # Compute and copy run-averaged RSA maps
        ttest_subject_rsa_runmean_map_name_list = []

        for subject_id in subject_list:
            subject_rsa_run_dir_path = (
                output_dir / subject_id / "rsa_map" / "feedback_model"
            )
            subject_rsa_run_map_name_list = [
                f"{subject_id}_{run_id}_task-photographer_{rsa_map_name}_rsa_correlation_map_rad{searchlight_radius}_blur{blur_kernel_width}.nii"
                for run_id in run_id_list
            ]
            subject_rsa_runmean_map_name = f"{subject_id}_task-photographer_{rsa_map_name}_within_run_mean_rsa_correlation_map_rad{searchlight_radius}_blur{blur_kernel_width}.nii"

            os.chdir(subject_rsa_run_dir_path)

            try:
                subprocess.run(
                    f"3dMean -overwrite -prefix {subject_rsa_runmean_map_name} {' '.join(subject_rsa_run_map_name_list)}",
                    shell=True,
                )
            except Exception as e:
                print(e)
                raise RuntimeError("Computation of within-run mean RSA map failed.")

            ttest_subject_rsa_runmean_map_name_list.append(subject_rsa_runmean_map_name)

            try:
                shutil.copy(
                    subject_rsa_run_dir_path / subject_rsa_runmean_map_name,
                    stat_ttest_dir,
                )
            except OSError:
                raise RuntimeError(
                    f"Cannot copy within-run mean RSA map ({rsa_map_name}) for {subject_id} ({subject_rsa_runmean_map_name}) in <{stat_ttest_dir}>"
                )

            # Sanity check
            original_rsa_map_array = load_nifti(
                subject_rsa_run_dir_path / subject_rsa_runmean_map_name
            ).data
            copied_rsa_map_array = load_nifti(
                stat_ttest_dir / subject_rsa_runmean_map_name
            ).data
            assert np.array_equal(original_rsa_map_array, copied_rsa_map_array) is True
            del original_rsa_map_array
            del copied_rsa_map_array

        os.chdir(stat_ttest_dir)

        # Run 3dttest++
        try:
            subprocess.run(
                f"3dttest++ -setA {' '.join(ttest_subject_rsa_runmean_map_name_list)} -mask {mni_gm_mask_path.name} -prefix feedback_rsa_ttest_{rsa_map_name}_within_run_mean_rad{searchlight_radius}_blur{blur_kernel_width}.nii -Clustsim",
                shell=True,
            )
        except Exception as e:
            print(e)
            raise RuntimeError(f"T-test of {rsa_map_name} RSA map failed.")

        print(f"T-test of {rsa_map_name} RSA map finished.")
        time.sleep(3)
