import os
import shutil
from pathlib import Path

from nipype.interfaces import afni

from ..utils.path import get_fmriprep_output_dir
from ..utils.types import ConfigDict


def prepare_gm_mask(config: ConfigDict):
    # `prepare_gm_mask` will use --participant-label to get the master brainmask data for resampling.
    if (
        config["execution"]["participant_label"] is None
        or len(config["execution"]["participant_label"]) > 1
    ):
        raise RuntimeError(
            "Need to specify only one (1) participant in --participant-label."
        )

    master_subject_id = f'sub-{config["execution"]["participant_label"][0]}'

    mni_gm_template_path = Path(config["execution"]["mask"]["mni_gm_template_path"])
    gm_probability_threshold = config["execution"]["mask"]["gm_probability_threshold"]

    # Check mni_gm_template_path is present
    if not mni_gm_template_path.exists():
        raise RuntimeError(f"Cannot find mni_gm_template: <{mni_gm_template_path}>")

    # Check fMRIPrep brainmask data of a specified subject is present
    fmriprep_output_dir = get_fmriprep_output_dir(config)
    master_subject_fmriprep_func_dir = fmriprep_output_dir / master_subject_id / "func"
    assert master_subject_fmriprep_func_dir.exists(), f"fMRIPrep output - func directory is not found: <{master_subject_fmriprep_func_dir}>"
    master_subject_brainmask_path = (
        master_subject_fmriprep_func_dir
        / f"{master_subject_id}_task-photographer_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    )

    if not master_subject_brainmask_path.exists():
        raise RuntimeError(
            f"fMRIPrep preprocessed brainmask is not found: <{master_subject_brainmask_path}>"
        )

    # Create output_dir/mask directory
    output_dir = Path(config["execution"]["output_dir"])
    assert output_dir.exists(), f"Output directory is not found: <{output_dir}>"

    try:
        output_mask_dir = output_dir / "mask"
        os.makedirs(output_mask_dir, exist_ok=True)
    except OSError:
        f"Cannot create mask directory: <{output_mask_dir}>"

    # Copy MNI GM template to mask directory
    try:
        if not Path(output_mask_dir / mni_gm_template_path.name).exists():
            shutil.copy(mni_gm_template_path, output_mask_dir)
    except OSError:
        raise RuntimeError(
            f"Cannot copy MNI GM template (<{mni_gm_template_path}>) to mask directory (<{output_mask_dir}>)."
        )

    # Copy brainmask data to mask directory
    try:
        shutil.copy(master_subject_brainmask_path, output_mask_dir)
    except OSError:
        raise RuntimeError(
            f"Cannot copy brainmask data (<{master_subject_brainmask_path}>) to mask directory (<{output_mask_dir}>)."
        )

    os.chdir(output_mask_dir)

    # Resample brainmask
    brainmask_resample_name = (
        f"{master_subject_id}_task-photographer_run-01_brainmask_resample.nii.gz"
    )
    resample = afni.Resample()
    resample.inputs.in_file = master_subject_brainmask_path.name
    resample.inputs.out_file = brainmask_resample_name
    resample.inputs.voxel_size = (3.0, 3.0, 3.0)
    resample.inputs.args = "-overwrite"
    resample.run()

    # Compute GM mask
    gm_mask_1mm_name = "mni_152_gm_mask_1mm.nii"
    calc = afni.Calc()
    calc.inputs.in_file_a = mni_gm_template_path.name
    calc.inputs.expr = f"ispositive(a-{gm_probability_threshold})"
    calc.inputs.out_file = gm_mask_1mm_name
    calc.inputs.args = "-overwrite"
    calc.run()

    gm_mask_3mm_name = "mni_152_gm_mask_3mm.nii"
    resample = afni.Resample()
    resample.inputs.in_file = gm_mask_1mm_name
    resample.inputs.out_file = gm_mask_3mm_name
    resample.inputs.master = brainmask_resample_name
    resample.inputs.args = "-overwrite"
    resample.run()

    # Remove unnecessary files
    Path(output_mask_dir / mni_gm_template_path.name).unlink(missing_ok=True)
    Path(output_mask_dir / master_subject_brainmask_path.name).unlink(missing_ok=True)
    Path(output_mask_dir / brainmask_resample_name).unlink(missing_ok=True)
    Path(output_mask_dir / gm_mask_1mm_name).unlink(missing_ok=True)

    print("Finished computing the MNI152 GM mask (3 mm)")
