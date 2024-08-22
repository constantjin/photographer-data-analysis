import os
from pathlib import Path

import pandas as pd
from numpy import format_float_positional

from ..utils.path import get_fmriprep_output_dir
from ..utils.subject_exclusion import delete_marked_subjects, mark_subject_exclusion
from ..utils.types import ConfigDict


def _subject_confound(subject_id: str, fmriprep_output_dir: Path, config: ConfigDict):
    output_dir = Path(config["execution"]["output_dir"])
    assert output_dir.exists(), f"Output directory is not found: <{output_dir}>"

    subject_fmriprep_func_dir = fmriprep_output_dir / subject_id / "func"
    assert (
        subject_fmriprep_func_dir.exists()
    ), f"fMRIPrep output - func directory is not found: <{subject_fmriprep_func_dir}>"

    subject_confounds_file_path_list = [
        child
        for child in subject_fmriprep_func_dir.iterdir()
        if child.name.endswith("confounds_timeseries.tsv")
    ]

    if len(subject_confounds_file_path_list) != 5:
        # Incomplete run, this subject needs to be excluded
        print(f"Incomplete run!: {subject_id}")
        mark_subject_exclusion(
            subject_id, f"Incomplte runs ({len(subject_confounds_file_path_list)})"
        )
        return

    marked_exclusion = False

    for run_confounds_file_path in subject_confounds_file_path_list:
        run_id = run_confounds_file_path.name.split("_")[2]
        print(subject_id, run_id)

        # Read confounds_timeseries.tsv
        run_confounds_df = pd.read_csv(run_confounds_file_path, sep="\t")

        # Check spike outlier columns first
        run_outliers_columns = [col for col in run_confounds_df if "outlier" in col]
        run_outlier_list = (
            run_confounds_df[run_outliers_columns]
            .apply(lambda row: row.max(), axis=1)
            .tolist()
        )
        run_outlier_volume_count = sum(run_outlier_list)
        run_outlier_volume_ratio = run_outlier_volume_count / len(run_outlier_list)

        if (
            run_outlier_volume_ratio
            > config["execution"]["glm"]["run_outlier_ratio_threshold"]
        ):
            print(
                f'Outlier censored run! (threshold = {int(config["execution"]["glm"]["run_outlier_ratio_threshold"] * 100)}%): '
                + f"{subject_id}, {run_id}, {round(run_outlier_volume_ratio * 100, 3)}%"
            )
            mark_subject_exclusion(
                subject_id,
                f"Outlier censored run ({run_id}, {round(run_outlier_volume_ratio * 100, 3)}%)",
                config,
            )
            marked_exclusion = True

        # If this subject is marked as excluded, skip all regressor extractions
        if marked_exclusion:
            continue

        # Create subject-id/run-id/regressors directory
        try:
            run_regressors_dir = output_dir / subject_id / run_id / "regressors"
            os.makedirs(run_regressors_dir, exist_ok=True)
        except OSError:
            raise RuntimeError(
                f"Cannot create regressors directory: <{run_regressors_dir}>"
            )

        # Extract confound regressors
        for confound_column in config["execution"]["glm"]["confound_list"]:
            confound_list = run_confounds_df[confound_column].tolist()

            # if the confound is derivative (or framewise displacement), replace NaN (at the first volume) to zero.
            if "derivative" in confound_column:
                confound_list[0] = 0.0

            if "framewise" in confound_column:
                confound_list[0] = 0.0

            try:
                with open(
                    run_regressors_dir
                    / f"{subject_id}_task-photographer_{run_id}_confound_{confound_column}.1D",
                    "w",
                ) as f:
                    f.writelines("\n".join(map(format_float_positional, confound_list)))
            except OSError:
                raise RuntimeError(
                    f"Cannot write confound_{confound_column}.1D for {subject_id}, {run_id}."
                )

        # Save outlier regressor
        try:
            with open(
                run_regressors_dir
                / f"{subject_id}_task-photographer_{run_id}_confound_outlier.1D",
                "w",
            ) as f:
                f.writelines(
                    "\n".join(map(lambda v: "1" if v == 1 else "0", run_outlier_list))
                )
        except OSError:
            raise RuntimeError(
                f"Cannot write confound_outlier.1D for {subject_id}, {run_id}."
            )


def prepare_confound(config: ConfigDict):
    fmriprep_output_dir = get_fmriprep_output_dir(config)

    # List all appropriate subjects/participants without faulty participants after fMRIPrep
    subject_list = [
        child.stem
        for child in fmriprep_output_dir.iterdir()
        if child.is_dir()
        and "sub-" in child.stem
        and child.stem not in config["execution"]["glm"]["fmriprep_faulty_subject_list"]
    ]

    if config["execution"]["participant_label"] is not None:
        subject_list = [
            child
            for child in subject_list
            if child[4:] in config["execution"]["participant_label"]
        ]

    if not subject_list:
        raise RuntimeError(
            "No participant is selected. Please check --participant-label or BIDS root directory."
        )

    # Exclude faulty fMRIPrep subject(s) first
    for subject_id in config["execution"]["glm"]["fmriprep_faulty_subject_list"]:
        mark_subject_exclusion(subject_id, "fMRIPrep faulty subject", config)

    # Iterate through all appropriate subjects
    for subject_id in subject_list:
        _subject_confound(subject_id, fmriprep_output_dir, config)

    # Delete excluded subjects
    delete_marked_subjects(config)
