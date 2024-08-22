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

"""
Block-wise GLM (GLM1) for the univariate analysis: 7 task block regressors + 1 parametric regressor (feedback score) 
"""


def __subject_run_block_wise_glm(
    subject_id: str, run_id: str, fmriprep_output_dir: Path, config: ConfigDict
):
    gc.collect()

    output_dir = Path(config["execution"]["output_dir"])
    assert output_dir.exists(), f"Output directory is not found: <{output_dir}>"

    subject_fmriprep_func_dir = fmriprep_output_dir / subject_id / "func"
    assert (
        subject_fmriprep_func_dir.exists()
    ), f"fMRIPrep output - func directory is not found: <{subject_fmriprep_func_dir}>"

    run_fmriprep_bold_path = (
        subject_fmriprep_func_dir
        / f"{subject_id}_task-photographer_{run_id}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    run_fmriprep_brainmask_path = (
        subject_fmriprep_func_dir
        / f"{subject_id}_task-photographer_{run_id}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    )

    # Check presence of bold and brainmask data
    if not run_fmriprep_bold_path.exists():
        raise RuntimeError(
            f"fMRIPrep preprocessed BOLD data is not found: <{run_fmriprep_bold_path}>"
        )

    if not run_fmriprep_brainmask_path.exists():
        raise RuntimeError(
            f"fMRIPrep preprocessed brainmask is not found: <{run_fmriprep_brainmask_path}>"
        )

    print(subject_id, run_id)

    # Create run GLM directory
    try:
        run_glm_block_dir = output_dir / subject_id / run_id / "glm_block_wise"
        os.makedirs(run_glm_block_dir, exist_ok=True)
    except OSError:
        raise RuntimeError(f"Cannot create glm_trial directory: <{run_glm_block_dir}>")

    # Copy preprocessed BOLD and brainmask NIFTI files
    run_glm_bold_path = run_glm_block_dir / run_fmriprep_bold_path.name
    run_glm_brainmask_path = run_glm_block_dir / run_fmriprep_brainmask_path.name

    try:
        if not run_glm_bold_path.exists():
            shutil.copy(run_fmriprep_bold_path, run_glm_block_dir)

        if not run_glm_brainmask_path.exists():
            shutil.copy(run_fmriprep_brainmask_path, run_glm_block_dir)
    except OSError:
        raise RuntimeError(
            f"Cannot copy fMRIPrep BOLD and brainmask data (from <{subject_fmriprep_func_dir}> "
            + f"to glm_trial directory <{run_glm_block_dir}>."
        )

    # Copy regressors directory
    try:
        run_glm_regressors_dir = output_dir / subject_id / run_id / "regressors"
        shutil.copytree(
            run_glm_regressors_dir, run_glm_block_dir / "regressors", dirs_exist_ok=True
        )
    except OSError:
        raise RuntimeError(
            f"Cannot copy regressors directory (<{run_glm_regressors_dir}>) to glm_trial directory (<{run_glm_block_dir}>)."
        )

    # Copy AFNI MNI152-SSW template
    try:
        afni_template_path = (
            Path(config["execution"]["glm"]["afni_path"])
            / "MNI152_2009_template_SSW.nii.gz"
        )
        if not (run_glm_block_dir / afni_template_path.name).exists():
            shutil.copy(afni_template_path, run_glm_block_dir)
    except OSError:
        raise RuntimeError(
            f"Cannot copy AFNI template (from <{afni_template_path}>) to glm_trial directory (<{run_glm_block_dir}>)."
        )

    os.chdir(run_glm_block_dir)

    # Resample 3 x 3 x 4 mm data to isotropic 3 mm data
    run_glm_bold_resample_name = (
        f"{subject_id}_task-photographer_{run_id}_bold_resample.nii.gz"
    )
    resample = afni.Resample()
    resample.inputs.in_file = run_glm_bold_path.name
    resample.inputs.out_file = run_glm_bold_resample_name
    resample.inputs.voxel_size = (3.0, 3.0, 3.0)
    resample.inputs.args = "-overwrite"
    resample.run()

    run_glm_brainmask_resample_name = (
        f"{subject_id}_task-photographer_{run_id}_brainmask_resample.nii.gz"
    )
    resample = afni.Resample()
    resample.inputs.in_file = run_glm_brainmask_path.name
    resample.inputs.out_file = run_glm_brainmask_resample_name
    resample.inputs.voxel_size = (3.0, 3.0, 3.0)
    resample.inputs.args = "-overwrite"
    resample.run()

    # Blur bold data
    blur_kernel_width = (
        config["execution"]["glm"]["glm_block_blur_kernel_width"]
        if config["execution"]["glm"]["glm_block_blur_kernel_width"]
        else 8
    )
    run_glm_bold_blur_name = (
        f"{subject_id}_task-photographer_{run_id}_bold_blur{blur_kernel_width}.nii"
    )
    merge = afni.Merge()
    merge.inputs.in_files = [run_glm_bold_resample_name]
    merge.inputs.blurfwhm = blur_kernel_width
    merge.inputs.doall = True
    merge.inputs.out_file = run_glm_bold_blur_name
    merge.inputs.args = "-overwrite"
    merge.run()

    # Scale bold data
    run_glm_bold_mean_name = f"{subject_id}_task-photographer_{run_id}_bold_mean.nii.gz"
    tstat = afni.TStat()
    tstat.inputs.in_file = run_glm_bold_blur_name
    tstat.inputs.out_file = run_glm_bold_mean_name
    tstat.inputs.args = "-overwrite"
    tstat.run()

    run_glm_bold_scale_name = (
        f"{subject_id}_task-photographer_{run_id}_bold_scale.nii.gz"
    )
    calc = afni.Calc()
    calc.inputs.in_file_a = run_glm_bold_blur_name
    calc.inputs.in_file_b = run_glm_bold_mean_name
    calc.inputs.in_file_c = run_glm_brainmask_resample_name
    calc.inputs.expr = "c * min(200, a/b*100)"
    calc.inputs.out_file = run_glm_bold_scale_name
    calc.inputs.args = "-overwrite"
    calc.run()

    # Prepare GLM regressors
    run_glm_event_order_path = (
        run_glm_block_dir
        / "regressors"
        / f"{subject_id}_task-photographer_{run_id}_trial_event_order.1D"
    )
    try:
        with open(run_glm_event_order_path, "r") as f:
            run_glm_event_order_list = f.readlines()
            run_glm_event_order_list = [
                line.strip() for line in run_glm_event_order_list
            ]
    except IOError:
        raise (f"Cannot read trial_event_order.1D: <{run_glm_event_order_path}>")

    stim_index = 0
    run_glm_events_list = [
        "exploration",
        "capture",
        "capture_failed",
        "preview",
        "voice",
        "caption",
        "feedback",
    ]
    run_glm_regressors_list = []

    # Task regressors
    for event_type in run_glm_events_list:
        event_file = f"regressors/{subject_id}_task-photographer_{run_id}_block_{event_type}_event.1D"

        if (run_glm_block_dir / event_file).exists():
            stim_index += 1

            # For feedback events, use AM2 modulation
            if "feedback" == event_type:
                run_glm_regressors_list.append(
                    f"-stim_times_AM2 {stim_index} {event_file} 'dmBLOCK' -stim_label {stim_index} {event_type}"
                )
            else:
                run_glm_regressors_list.append(
                    f"-stim_times_AM1 {stim_index} {event_file} 'dmBLOCK' -stim_label {stim_index} {event_type}"
                )

    # Nuisance regressors
    for confound_label in config["execution"]["glm"]["confound_list"]:
        stim_index += 1
        confound_file = f"regressors/{subject_id}_task-photographer_{run_id}_confound_{confound_label}.1D"

        if not (run_glm_block_dir / confound_file).exists():
            raise RuntimeError(
                f"Confound file for {confound_label} does not exist: <{run_glm_block_dir / confound_file}>"
            )

        run_glm_regressors_list.append(
            f"-stim_file {stim_index} {confound_file} -stim_base {stim_index} -stim_label {stim_index} {confound_label}"
        )

    # Outlier volume regressor
    outlier_file = (
        f"regressors/{subject_id}_task-photographer_{run_id}_confound_outlier.1D"
    )

    if not (run_glm_block_dir / outlier_file).exists():
        raise RuntimeError(
            f"Outlier file does not exist: <{run_glm_block_dir / outlier_file}>"
        )

    # Check values in the outlier column are all zero
    try:
        with open(run_glm_block_dir / outlier_file, "r") as f:
            outlier_lines = [int(line.strip()) for line in f.readlines()]

            # If not, include the outlier column as a regressor
            if not all(v == 0 for v in outlier_lines):
                stim_index += 1
                run_glm_regressors_list.append(
                    f"-stim_file {stim_index} {outlier_file} -stim_base {stim_index} -stim_label {stim_index} outlier"
                )
    except IOError:
        raise RuntimeError(
            f"Cannot read outlier file: <{run_glm_block_dir / outlier_file}>"
        )

    # Run 3dDeconvolve
    run_glm_args = f'-num_stimts {stim_index} {" ".join(run_glm_regressors_list)} -xjpeg X.jpg -x1D_uncensored X.nocensor.xmat.1D -fitts fitts.{subject_id}.{run_id} -errts errts.{subject_id}.{run_id} -jobs 8 -overwrite'

    try:
        deconvolve = afni.Deconvolve()
        deconvolve.inputs.in_files = [run_glm_bold_scale_name]
        deconvolve.inputs.mask = run_glm_brainmask_resample_name
        deconvolve.inputs.stim_times_subtract = 2.0 / 2
        deconvolve.inputs.polort = 5
        deconvolve.inputs.local_times = True
        deconvolve.inputs.fout = True
        deconvolve.inputs.tout = True
        deconvolve.inputs.x1D = "X.xmat.1D"
        deconvolve.inputs.out_file = (
            f"{subject_id}_task-photographer_{run_id}_stats.nii"
        )
        deconvolve.inputs.args = run_glm_args
        deconvolve.run()
    except Exception as e:
        print(e)
        raise RuntimeError(f"GLM failed: {subject_id} {run_id}")

    # display any large pairwise correlations from the X-matrix
    try:
        subprocess.call(
            "1d_tool.py -show_cormat_warnings -infile X.xmat.1D > out.cormat_warn.txt",
            shell=True,
        )
    except Exception as e:
        print(e)
        raise RuntimeError("Failed to call 1d_tool.py.")

    print(f"GLM finished: {subject_id} {run_id}")


def run_block_wise_glm(config: ConfigDict):
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
        for run_id in ["run-01", "run-02", "run-03", "run-04", "run-05"]:
            __subject_run_block_wise_glm(
                subject_id, run_id, fmriprep_output_dir, config
            )
            time.sleep(2)
