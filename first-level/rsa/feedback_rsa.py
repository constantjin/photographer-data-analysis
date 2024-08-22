import gc
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from nipype.interfaces import afni
from rsatoolbox.rdm import compare_rho_a
from scipy.spatial.distance import pdist

from ..utils.nifti import NiftiImage, load_nifti, save_nifti
from ..utils.parallel import pmap
from ..utils.path import get_fmriprep_output_dir
from ..utils.searchlight import Searchlight
from ..utils.subject_exclusion import read_subject_exclusion
from ..utils.types import ConfigDict

"""
Feedback model RSA
- feedback model RDMs constructed from raw feedback scores (0 - 100 range)
- neural data from trial-wise GLM (GLM2)
- MNI152 GM mask (threshold = 0.3; 3 mm)
- rsatoolbox compare_rho_a
- run-wise
"""


def _load_rdm_from_numpy(
    data_dir: Path, prefix: str, subject_id: str, run_id: str, data_name: str
):
    try:
        numpy_path = (
            data_dir
            / prefix
            / f"{subject_id}_{run_id}_task-photographer_{data_name}_vector.npy"
        )
        return np.load(numpy_path)
    except IOError:
        raise RuntimeError(f"Cannot load numpy array: <{numpy_path}>")


def _create_neural_rdm_vector(sphere: tuple[np.ndarray, tuple[int, int, int]]):
    neural_vector, center_voxel_index = sphere
    neural_rdm_vector = pdist(neural_vector.T, "correlation")
    neural_rdm_vector = np.nan_to_num(
        neural_rdm_vector, nan=1.0
    )  # correlation distances can be nan!
    return (center_voxel_index, neural_rdm_vector)


def _generate_neural_rdm_sphere_list(
    sphere_list: list[tuple[np.ndarray, tuple[int, int, int]]],
):
    neural_rdm_sphere_list = pmap(
        _create_neural_rdm_vector, sphere_list, pm_pbar=True, pm_chunksize=500
    )

    # filter neural RDM spheres whose RDM vector does not contain only one type of value
    # sphere[1] -> neural RDM vector, sphere[1][0] -> first element of neural RDM vector
    filtered_neural_rdm_sphere_list = list(
        filter(
            lambda sphere: not np.all((sphere[1] == sphere[1][0])),
            neural_rdm_sphere_list,
        )
    )

    return filtered_neural_rdm_sphere_list


def _sphere_level_correlation(
    neural_rdm_sphere: tuple[tuple[int, int, int], np.ndarray],
    model_rdm_vector: np.ndarray,
):
    center_voxel_index, neural_rdm_vector = neural_rdm_sphere
    raw_corr_coef = compare_rho_a(neural_rdm_vector, model_rdm_vector)[0][0]
    zscored_corr_coef = np.arctanh(raw_corr_coef)

    return (center_voxel_index, zscored_corr_coef)


def _compute_neural_model_correlation_map(
    neural_rdm_sphere_list: list[tuple[tuple[int, int, int], np.ndarray]],
    model_rdm_vector: np.ndarray,
    dim,
):
    rsa_output_brain_map = np.zeros(dim)

    rsa_output_correlation_list = pmap(
        _sphere_level_correlation,
        neural_rdm_sphere_list,
        model_rdm_vector,
        pm_pbar=True,
        pm_chunksize=300,
    )

    for center_voxel_index, model_correlation_coef in rsa_output_correlation_list:
        rsa_output_brain_map[center_voxel_index] = model_correlation_coef

    return rsa_output_brain_map


def _save_and_blur_nifti_rsa_map(
    brain_map: np.ndarray,
    template_nifti: NiftiImage,
    result_dir: Path,
    subject_id: str,
    run_id: str,
    map_name: str,
    searchlight_radius: int,
    blur_kernel_width: int,
):
    save_nifti(
        brain_map,
        template_nifti,
        result_dir,
        f"{subject_id}_{run_id}_task-photographer_{map_name}_rsa_correlation_map_rad{searchlight_radius}",
    )

    # Modify qform/sform header to 4 (MNI space)
    try:
        subprocess.run(
            f"nifti_tool -mod_hdr -overwrite -infiles {subject_id}_{run_id}_task-photographer_{map_name}_rsa_correlation_map_rad{searchlight_radius}.nii -mod_field qform_code 4 -mod_field sform_code 4",
            shell=True,
        )
    except Exception as e:
        print(e)
        raise RuntimeError("Modification of the nifti header failed.")

    merge = afni.Merge()
    merge.inputs.in_files = [
        f"{subject_id}_{run_id}_task-photographer_{map_name}_rsa_correlation_map_rad{searchlight_radius}.nii"
    ]
    merge.inputs.blurfwhm = blur_kernel_width
    merge.inputs.doall = True
    merge.inputs.out_file = f"{subject_id}_{run_id}_task-photographer_{map_name}_rsa_correlation_map_rad{searchlight_radius}_blur{blur_kernel_width}.nii"
    merge.inputs.args = "-overwrite"
    merge.run()


def _perform_individual_rsa(subject_id: str, config: ConfigDict):
    gc.collect()

    print(subject_id)

    output_dir = Path(config["execution"]["output_dir"])
    assert output_dir.exists(), f"Output directory is not found: <{output_dir}>"

    rsa_model_rdm_dir = output_dir / subject_id / "rsa_model_rdm"
    if not rsa_model_rdm_dir.exists():
        raise RuntimeError(f"Model RDM directory not found: <{rsa_model_rdm_dir}>")

    rsa_neural_data_dir = output_dir / subject_id / "rsa_neural_data"
    if not rsa_neural_data_dir.exists():
        raise RuntimeError(f"Neural data directory not found: <{rsa_neural_data_dir}>")

    mni_152_gm_mask_path = output_dir / "mask" / "mni_152_gm_mask_3mm.nii"
    if not mni_152_gm_mask_path.exists():
        raise RuntimeError(f"MNI152 GM mask not found: <{mni_152_gm_mask_path}>")

    try:
        mni_152_gm_mask_image = load_nifti(
            mni_152_gm_mask_path, save_dim=True, save_affine=True
        )
    except Exception as e:
        print(e)
        raise RuntimeError(
            f"Cannot load MNI152 GM mask image: <{mni_152_gm_mask_path}>"
        )

    # Create RSA output directory
    try:
        rsa_result_dir = output_dir / subject_id / "rsa_map" / "feedback_model"
        os.makedirs(rsa_result_dir, exist_ok=True)
    except OSError:
        raise RuntimeError(
            f"Cannot make individual RSA map directory: <{rsa_result_dir}>"
        )

    # Copy AFNI MNI152-SSW template
    try:
        afni_template_path = (
            Path(config["execution"]["glm"]["afni_path"])
            / "MNI152_2009_template_SSW.nii.gz"
        )
        if not (rsa_result_dir / afni_template_path.name).exists():
            shutil.copy(afni_template_path, rsa_result_dir)
    except OSError:
        raise RuntimeError(
            f"Cannot copy AFNI template (from <{afni_template_path}>) to RSA map directory (<{rsa_result_dir}>)."
        )

    os.chdir(rsa_result_dir)

    rsa_feedback_model_name_list = [
        "current_trial",
        "one_back_trial",
        "two_back_trial",
        "recent_2_trial",
        "recent_3_trial",
        "previous_2_trial",
    ]
    run_id_list = ["run-01", "run-02", "run-03", "run-04", "run-05"]

    for run_id in run_id_list:
        # Check data paths
        rsa_feedback_neural_data_path = (
            rsa_neural_data_dir
            / "feedback_beta"
            / f"{subject_id}_{run_id}_task-photographer_trial_feedback_norm_beta_array.npy"
        )
        if not rsa_feedback_neural_data_path.exists():
            raise RuntimeError(
                f'Feedback neural data numpy array not found: <{rsa_feedback_neural_data_path}>. Please run "rsa.prepare_feedback_neural_data" task first'
            )

        # Load neural/model data
        try:
            rsa_trial_feedback_norm_beta_array = np.load(rsa_feedback_neural_data_path)
        except IOError:
            raise RuntimeError(
                f"Cannot load trial_feedback_norm_beta_array.npy: <{rsa_feedback_neural_data_path}>"
            )

        rsa_feedback_model_vector_list = []
        try:
            for rsa_model_name in rsa_feedback_model_name_list:
                rsa_model_array = _load_rdm_from_numpy(
                    rsa_model_rdm_dir,
                    "feedback_model",
                    subject_id,
                    run_id,
                    rsa_model_name,
                )
                rsa_feedback_model_vector_list.append(rsa_model_array)
        except RuntimeError as e:
            print(e)
            raise RuntimeError(
                f'Cannot load feedback model RDM: {rsa_model_name} for {subject_id}. Please run "rsa.prepare_feedback_model_rdm" task first.'
            )

        # compute neural searchlight sphere list
        searchlight_radius = (
            config["execution"]["rsa"]["searchlight_radius"]
            if config["execution"]["rsa"]["searchlight_radius"]
            else 3
        )
        searchlight = Searchlight(searchlight_radius)
        rsa_feedback_neural_searchlight_sphere_list = [
            inform
            for inform in searchlight.analysis(
                data=rsa_trial_feedback_norm_beta_array, mask=mni_152_gm_mask_image.data
            )
        ]
        print(f"Computed neural searchlight sphere list: radius = {searchlight_radius}")

        # compute neural RDM sphere list
        rsa_feedback_neural_rdm_sphere_list = _generate_neural_rdm_sphere_list(
            rsa_feedback_neural_searchlight_sphere_list
        )
        print(
            f"Computed neural RDM sphere list: shape = {rsa_feedback_neural_rdm_sphere_list[0][1].shape}"
        )

        # perform actual RSA
        blur_kernel_width = (
            config["execution"]["rsa"]["rsa_blur_kernel_width"]
            if config["execution"]["rsa"]["rsa_blur_kernel_width"]
            else 6
        )
        for rsa_feedback_model_name, rsa_feedback_model_vector in zip(
            rsa_feedback_model_name_list, rsa_feedback_model_vector_list
        ):
            print(f"Computing {rsa_feedback_model_name} RSA map")
            rsa_brain_map = _compute_neural_model_correlation_map(
                rsa_feedback_neural_rdm_sphere_list,
                rsa_feedback_model_vector,
                mni_152_gm_mask_image.dim,
            )

            _save_and_blur_nifti_rsa_map(
                rsa_brain_map,
                mni_152_gm_mask_image,
                rsa_result_dir,
                subject_id,
                run_id,
                f"{rsa_feedback_model_name}",
                searchlight_radius,
                blur_kernel_width,
            )

            print(f"Saved {rsa_feedback_model_name} RSA map.")


def run_feedback_rsa(config: ConfigDict):
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
        _perform_individual_rsa(subject_id, config)
        time.sleep(2)
