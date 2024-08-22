from pathlib import Path
from typing import Optional, TypedDict


# Type annotations for the config toml file
class GLMConfigDict(TypedDict):
    fmriprep_version: str  # fMRIPrep version
    performed_reconall: bool  # Whether you ran the fMRIPrep with (false) or without (true) the `--fs-no-reconall`` option
    fmriprep_faulty_subject_list: list[
        str
    ]  # Specify subject ID(s) whose fMRIPrep derivatives look faulty
    confound_list: list[
        str
    ]  # List of confound variables to be included as nuisance regressors
    behavioral_data_dir: (
        Path | str
    )  # Path to the behavioral data directory from the actual Photographer paradigm
    run_outlier_ratio_threshold: (
        float  # (# of outlier volumes) / (total volumes) of runs to be excluded
    )
    afni_path: Path | str  # Path to the AFNI `abin` directory
    glm_block_blur_kernel_width: (
        int  # Smoothing Gaussian kernel FWHM for block-wise GLM
    )


class MaskConfigDict(TypedDict):
    mni_gm_template_path: (
        Path | str
    )  # Full path to the MNI152NLin2009cAsym templete nii file
    gm_probability_threshold: float  # Gray matter probability threshold for the GM mask


class RSAConfigDict(TypedDict):
    univariate_noise_normalization: bool  # Whether or not to apply the univariate noise normalization to beta values
    searchlight_radius: int  # Searchlight kernel radius in voxels
    rsa_blur_kernel_width: int  # Smoothing Gaussian kernel FWHM on the raw RSA maps


class ExecutionConfigDict(TypedDict):
    glm: GLMConfigDict
    mask: MaskConfigDict
    rsa: RSAConfigDict

    bids_dir: Path
    output_dir: Path
    analysis_level: str
    participant_label: Optional[list[str]]
    task: str
    config_file: Path

    subject_exclusion_file_path: Path


class ConfigDict(TypedDict):
    execution: ExecutionConfigDict


# Type annotations for the subject_exclusion.json file
ExclusionDict = dict[str, list[str]]
