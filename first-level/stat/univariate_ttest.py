import gc
import os
import shutil
import subprocess
import time
from pathlib import Path

from nipype.interfaces import afni

from ..utils.path import get_fmriprep_output_dir
from ..utils.subject_exclusion import read_subject_exclusion
from ..utils.types import ConfigDict


def run_univariate_ttest(config: ConfigDict):
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

    # Create stat - ttest directory
    try:
        stat_ttest_dir = output_dir / "stat" / "univariate" / "block_wise" / "ttest"
        os.makedirs(stat_ttest_dir, exist_ok=True)
    except OSError:
        raise RuntimeError(f"Cannot create t-test output directory: <{stat_ttest_dir}>")

    # Aggregate GLM output files and check presence
    run_id_list = ["run-01", "run-02", "run-03", "run-04", "run-05"]
    glm_block_1_path_list = [
        [
            output_dir
            / subject_id
            / run_id
            / "glm_block_wise"
            / f"{subject_id}_task-photographer_{run_id}_stats.nii"
            for run_id in run_id_list
        ]
        for subject_id in subject_list
    ]

    for subject_path_list in glm_block_1_path_list:
        for run_path in subject_path_list:
            if not run_path.exists():
                raise RuntimeError(f"GLM block 1 stat.nii not found: <{run_path}>")

    block_regressor_list = [
        "exploration",
        "capture",
        "capture_failed",
        "preview",
        "voice",
        "caption",
        "feedback",
    ]
    # Process block (task) regressors
    for block_regressor in block_regressor_list:
        gc.collect()
        print("Block regressor:", block_regressor)

        try:
            ttest_block_dir = stat_ttest_dir / f"block_{block_regressor}"
            os.makedirs(ttest_block_dir, exist_ok=True)
        except OSError:
            raise RuntimeError(
                f"Cannot create t-test block regressor output directory: <{ttest_block_dir}>"
            )

        os.chdir(ttest_block_dir)

        for subject_id in subject_list:
            subject_block_beta_map_name_list = []

            # Extract block regressor beta map from each run
            for run_id in run_id_list:
                run_glm_stat_data_path = (
                    output_dir
                    / subject_id
                    / run_id
                    / "glm_block_wise"
                    / f"{subject_id}_task-photographer_{run_id}_stats.nii"
                )
                subbrick_identifier = f"[{block_regressor}#0_Coef]"
                ttest_run_block_beta_map_name = f"{subject_id}_task-photographer_{run_id}_block_{block_regressor}_beta.nii"

                bucket = afni.Bucket()
                bucket.inputs.in_file = [(run_glm_stat_data_path, subbrick_identifier)]
                bucket.inputs.out_file = ttest_run_block_beta_map_name
                bucket.inputs.args = "-overwrite"
                bucket.run()

                if (ttest_block_dir / ttest_run_block_beta_map_name).exists():
                    subject_block_beta_map_name_list.append(
                        ttest_run_block_beta_map_name
                    )

            # Mean all block regressor beta map
            if subject_block_beta_map_name_list:
                subprocess.run(
                    f"3dMean -overwrite -prefix {subject_id}_task-photographer_mean_block_{block_regressor}_beta.nii {' '.join(subject_block_beta_map_name_list)}",
                    shell=True,
                )

        # Copy GM mask and MNI template
        try:
            mni_gm_mask_path = output_dir / "mask" / "mni_152_gm_mask_3mm.nii"
            if not (ttest_block_dir / mni_gm_mask_path.name).exists():
                shutil.copy(mni_gm_mask_path, ttest_block_dir)
        except OSError:
            raise RuntimeError(
                f"Cannot copy MNI152 GM mask (from <{mni_gm_mask_path}>) to ttest block directory (<{ttest_block_dir}>)."
            )

        try:
            afni_template_path = (
                Path(config["execution"]["glm"]["afni_path"])
                / "MNI152_2009_template_SSW.nii.gz"
            )
            if not (ttest_block_dir / afni_template_path.name).exists():
                shutil.copy(afni_template_path, ttest_block_dir)
        except OSError:
            raise RuntimeError(
                f"Cannot copy AFNI template (from <{afni_template_path}>) to ttest block directory (<{ttest_block_dir}>)."
            )

        # Run 3dttest++
        try:
            subprocess.run(
                f"3dttest++ -setA '*mean_block_{block_regressor}_beta.nii' -mask {mni_gm_mask_path.name} -prefix univariate_ttest_block_{block_regressor}.nii -Clustsim",
                shell=True,
            )
        except Exception as e:
            print(e)
            raise RuntimeError(f"T-test of block regressor {block_regressor} failed.")

        print(f"T-test of block regressor {block_regressor} finished.")
        time.sleep(2)

    parametric_regressor_list = ["feedback"]
    # Process parametric regressor(s)
    for parametric_regressor in parametric_regressor_list:
        gc.collect()
        print("Parametric regressor:", parametric_regressor)

        try:
            ttest_parametric_dir = stat_ttest_dir / f"parametric_{parametric_regressor}"
            os.makedirs(ttest_parametric_dir, exist_ok=True)
        except OSError:
            raise RuntimeError(
                f"Cannot create t-test parametric regressor output directory: <{ttest_parametric_dir}>"
            )

        os.chdir(ttest_parametric_dir)

        for subject_id in subject_list:
            subject_parametric_beta_map_name_list = []

            # Extract parametric regressor beta map from each run
            for run_id in run_id_list:
                run_glm_stat_data_path = (
                    output_dir
                    / subject_id
                    / run_id
                    / "glm_block_wise"
                    / f"{subject_id}_task-photographer_{run_id}_stats.nii"
                )
                subbrick_identifier = f"[{parametric_regressor}#1_Coef]"
                ttest_run_parametric_beta_map_name = f"{subject_id}_task-photographer_{run_id}_parametric_{parametric_regressor}_beta.nii"

                bucket = afni.Bucket()
                bucket.inputs.in_file = [(run_glm_stat_data_path, subbrick_identifier)]
                bucket.inputs.out_file = ttest_run_parametric_beta_map_name
                bucket.inputs.args = "-overwrite"
                bucket.run()

                if (ttest_parametric_dir / ttest_run_parametric_beta_map_name).exists():
                    subject_parametric_beta_map_name_list.append(
                        ttest_run_parametric_beta_map_name
                    )

            # Mean all parametric regressor beta map
            if subject_parametric_beta_map_name_list:
                subprocess.run(
                    f"3dMean -overwrite -prefix {subject_id}_task-photographer_mean_parametric_{parametric_regressor}_beta.nii {' '.join(subject_parametric_beta_map_name_list)}",
                    shell=True,
                )

        # Copy GM mask and MNI template
        try:
            mni_gm_mask_path = output_dir / "mask" / "mni_152_gm_mask_3mm.nii"
            if not (ttest_parametric_dir / mni_gm_mask_path.name).exists():
                shutil.copy(mni_gm_mask_path, ttest_parametric_dir)
        except OSError:
            raise RuntimeError(
                f"Cannot copy MNI152 GM mask (from <{mni_gm_mask_path}>) to ttest parametric directory (<{ttest_parametric_dir}>)."
            )

        try:
            afni_template_path = (
                Path(config["execution"]["glm"]["afni_path"])
                / "MNI152_2009_template_SSW.nii.gz"
            )
            if not (ttest_parametric_dir / afni_template_path.name).exists():
                shutil.copy(afni_template_path, ttest_parametric_dir)
        except OSError:
            raise RuntimeError(
                f"Cannot copy AFNI template (from <{afni_template_path}>) to ttest parametric directory (<{ttest_parametric_dir}>)."
            )

        # Run 3dttest++
        try:
            subprocess.run(
                f"3dttest++ -setA '*mean_parametric_{parametric_regressor}_beta.nii' -mask {mni_gm_mask_path.name} -prefix univariate_ttest_parametric_{parametric_regressor}.nii -Clustsim",
                shell=True,
            )
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"T-test of parametric regressor {parametric_regressor} failed."
            )

        print(f"T-test of parametric regressor {parametric_regressor} finished.")
