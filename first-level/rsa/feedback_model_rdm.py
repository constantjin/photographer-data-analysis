import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from ..utils.path import get_fmriprep_output_dir
from ..utils.subject_exclusion import read_subject_exclusion
from ..utils.types import ConfigDict


def _rdm_save_as_numpy(
    output_dir: Path,
    prefix: str,
    subject_id: str,
    run_id: str,
    data_name: str,
    rdm_vector: np.ndarray,
):
    try:
        os.makedirs(output_dir / prefix, exist_ok=True)
    except OSError:
        raise RuntimeError(
            f"Cannot create {prefix} directory in {output_dir} for {subject_id}"
        )

    try:
        numpy_path = (
            output_dir
            / prefix
            / f"{subject_id}_{run_id}_task-photographer_{data_name}_vector.npy"
        )
        np.save(numpy_path, rdm_vector)
    except IOError:
        raise RuntimeError(
            f"Cannot save {prefix}/{data_name} in numpy array <{numpy_path}>"
        )


def _compute_individual_run_feedback_model_rdm(
    subject_id: str, run_id: str, config: ConfigDict
):
    output_dir = Path(config["execution"]["output_dir"])

    behavioral_data_dir_path = Path(config["execution"]["glm"]["behavioral_data_dir"])
    subgroup = str(config["execution"]["bids_dir"]).split("/bids")[0].split("/")[-1]

    try:
        behavior_feedback_df_path = (
            behavioral_data_dir_path / f"group_{subgroup}_behavior_feedback.csv"
        )
        behavior_feedback_df = pd.read_csv(behavior_feedback_df_path)
    except IOError:
        raise RuntimeError(
            f"Behavior - Feedback data file cannot be loaded: <{behavior_feedback_df_path}>. Please run 'behavior.prepare_behavioral_data' command"
        )

    run_id_int = int(run_id.split("-")[-1][-1])

    subject_run_feedback_data_df = behavior_feedback_df[
        (behavior_feedback_df["subject_id"] == subject_id)
        & (behavior_feedback_df["run"] == run_id_int)
    ]

    all_trial_feedback_score_list = subject_run_feedback_data_df[
        "feedback_score"
    ].to_numpy()

    n_trials_to_use = 6  # up to 2-back trials

    run_current_trial_list = [
        [elem] for elem in all_trial_feedback_score_list[-n_trials_to_use:]
    ]
    assert len(run_current_trial_list) == n_trials_to_use

    run_one_back_trial_list = [
        [elem] for elem in all_trial_feedback_score_list[-n_trials_to_use - 1 : -1]
    ]

    run_two_back_trial_list = [
        [elem] for elem in all_trial_feedback_score_list[-n_trials_to_use - 2 : -2]
    ]

    run_recent_2_trial_list = [
        [
            all_trial_feedback_score_list[idx - 1],
            all_trial_feedback_score_list[idx],
        ]
        for idx in range(2, 8)
    ]

    run_recent_3_trial_list = [
        [
            all_trial_feedback_score_list[idx - 2],
            all_trial_feedback_score_list[idx - 1],
            all_trial_feedback_score_list[idx],
        ]
        for idx in range(2, 8)
    ]

    run_previous_2_trial_list = [
        [
            all_trial_feedback_score_list[idx - 2],
            all_trial_feedback_score_list[idx - 1],
        ]
        for idx in range(2, 8)
    ]

    # Compute feedback model rdm vectors
    feedback_current_trial_rdm_vector = pdist(np.array(run_current_trial_list))
    assert feedback_current_trial_rdm_vector.shape == (15,)  # 6 * (6 - 1) / 2 = 15

    feedback_one_back_trial_rdm_vector = pdist(np.array(run_one_back_trial_list))
    feedback_two_back_trial_rdm_vector = pdist(np.array(run_two_back_trial_list))

    feedback_recent_2_trial_rdm_vector = pdist(np.array(run_recent_2_trial_list))
    assert feedback_recent_2_trial_rdm_vector.shape == (15,)
    feedback_recent_3_trial_rdm_vector = pdist(np.array(run_recent_3_trial_list))
    feedback_previous_2_trial_rdm_vector = pdist(np.array(run_previous_2_trial_list))

    # create RSA model RDM result directory
    try:
        subject_model_rdm_dir = output_dir / subject_id / "rsa_model_rdm"
        os.makedirs(subject_model_rdm_dir, exist_ok=True)
    except OSError:
        raise RuntimeError(
            f"Cannot create rsa_model_rdm directory: <{subject_model_rdm_dir}>"
        )

    # save RDM/mask vectors into numpy arrays
    _rdm_save_as_numpy(
        subject_model_rdm_dir,
        "feedback_model",
        subject_id,
        run_id,
        "current_trial",
        feedback_current_trial_rdm_vector,
    )
    _rdm_save_as_numpy(
        subject_model_rdm_dir,
        "feedback_model",
        subject_id,
        run_id,
        "one_back_trial",
        feedback_one_back_trial_rdm_vector,
    )
    _rdm_save_as_numpy(
        subject_model_rdm_dir,
        "feedback_model",
        subject_id,
        run_id,
        "two_back_trial",
        feedback_two_back_trial_rdm_vector,
    )

    _rdm_save_as_numpy(
        subject_model_rdm_dir,
        "feedback_model",
        subject_id,
        run_id,
        "recent_2_trial",
        feedback_recent_2_trial_rdm_vector,
    )
    _rdm_save_as_numpy(
        subject_model_rdm_dir,
        "feedback_model",
        subject_id,
        run_id,
        "recent_3_trial",
        feedback_recent_3_trial_rdm_vector,
    )
    _rdm_save_as_numpy(
        subject_model_rdm_dir,
        "feedback_model",
        subject_id,
        run_id,
        "previous_2_trial",
        feedback_previous_2_trial_rdm_vector,
    )


def _extract_subject_feedback_model_rdm(subject_id: str, config):
    gc.collect()

    behavioral_data_dir = Path(config["execution"]["glm"]["behavioral_data_dir"])
    assert (
        behavioral_data_dir.exists()
    ), f"Behavioral data directory is not found: <{behavioral_data_dir}>"

    output_dir = Path(config["execution"]["output_dir"])
    assert output_dir.exists(), f"Output directory is not found: <{output_dir}>"

    print(subject_id)
    run_id_list = ["run-01", "run-02", "run-03", "run-04", "run-05"]

    for run_id in run_id_list:
        _compute_individual_run_feedback_model_rdm(subject_id, run_id, config)


def prepare_feedback_model_rdm(config):
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
        _extract_subject_feedback_model_rdm(subject_id, config)
