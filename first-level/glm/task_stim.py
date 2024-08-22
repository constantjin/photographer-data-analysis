import os
from datetime import datetime
from pathlib import Path

from ..utils.path import get_fmriprep_output_dir
from ..utils.subject_exclusion import delete_marked_subjects, mark_subject_exclusion
from ..utils.types import ConfigDict

MAX_REWARD = 0.28


def _subject_task_stim(subject_id: str, config: ConfigDict):
    behavioral_data_dir = Path(config["execution"]["glm"]["behavioral_data_dir"])
    assert (
        behavioral_data_dir.exists()
    ), f"Behavioral data directory is not found: <{behavioral_data_dir}>"

    output_dir = Path(config["execution"]["output_dir"])
    assert output_dir.exists(), f"Output directory is not found: <{output_dir}>"

    try:
        subject_behavior_dir = [
            child
            for child in behavioral_data_dir.iterdir()
            if subject_id.split("-")[1] in child.stem
        ][0]
    except IndexError:
        raise RuntimeError(f"Behavioral data directory for {subject_id} is not found.")

    subject_etime_path_list = [
        run_dir / "log_etime.txt"
        for run_dir in subject_behavior_dir.iterdir()
        if run_dir.is_dir() and "backup" not in run_dir.stem
    ]

    if len(subject_etime_path_list) != 5:
        # Incomplete run, this subject needs to be excluded
        print(f"Incomplete run!: {subject_id}")
        mark_subject_exclusion(
            subject_id, f"Incomplte runs ({len(subject_etime_path_list)})"
        )
        return

    marked_exclusion = False

    for run_etime_path in subject_etime_path_list:
        run_id = f'run-0{run_etime_path.parent.name.split("_")[0]}'
        print(subject_id, run_id)

        try:
            with open(run_etime_path, "r") as f:
                run_etime_lines = f.readlines()
        except IOError:
            raise RuntimeError(f"Cannot read etime: <{run_etime_path}>")

        run_parsed_lines = list(map(lambda line: line.split("\t"), run_etime_lines))
        run_parsed_lines = list(
            map(
                lambda t: (datetime.strptime(t[0], "%Y-%m-%d %H:%M:%S.%f"), t[1]),
                run_parsed_lines,
            )
        )
        run_start_time = list(run_parsed_lines)[0][0]
        run_parsed_lines = list(
            map(
                lambda t: ((t[0] - run_start_time).total_seconds(), t[1]),
                run_parsed_lines,
            )
        )

        trial_index = 1
        exploration_onset = 0

        # Trial-wise events
        trial_event_list = []

        # Block-wise events
        block_exploration_event_list = []
        block_capture_event_list = []
        block_capture_failed_event_list = []
        block_preview_event_list = []
        block_voice_event_list = []
        block_caption_event_list = []
        block_feedback_event_list = []

        for timestamp, message in run_parsed_lines:
            # exploration onset
            if f"trial_{trial_index}" in message:
                exploration_onset = timestamp

            # exploration duration, capture onset + duration (0.1 s)
            elif "capture" in message:
                exploration_duration = round(timestamp - exploration_onset, 3)
                trial_event_list.append(
                    (
                        f"trial{trial_index}_exploration",
                        f"{exploration_onset}:{exploration_duration}",
                    )
                )
                block_exploration_event_list.append(
                    f"{exploration_onset}:{exploration_duration}"
                )

                if "capture_failed" in message:
                    trial_event_list.append(
                        (f"trial{trial_index}_capture_failed", f"{timestamp}:0.1")
                    )
                    block_capture_failed_event_list.append(f"{timestamp}:0.1")
                else:
                    trial_event_list.append(
                        (f"trial{trial_index}_capture", f"{timestamp}:0.1")
                    )
                    block_capture_event_list.append(f"{timestamp}:0.1")

            # preview onset + duration (2 s)
            elif "trial_preview" in message:
                trial_event_list.append(
                    (f"trial{trial_index}_preview", f"{timestamp}:2")
                )
                block_preview_event_list.append(f"{timestamp}:2")

            # voice onset + duration (3 s)
            elif "trial_voice" in message:
                trial_event_list.append((f"trial{trial_index}_voice", f"{timestamp}:3"))
                block_voice_event_list.append(f"{timestamp}:3")

            # caption onset + duration (3 s)
            elif "trial_caption" in message:
                trial_event_list.append(
                    (f"trial{trial_index}_caption", f"{timestamp}:3")
                )
                block_caption_event_list.append(f"{timestamp}:3")

            # reward onset + duration (2 s)
            elif "trial_reward" in message:
                trial_raw_score = float(message.split("/percent:")[-1]) / 100
                trial_current_sim = float(message.split("/")[0].split(":")[-1])
                if trial_current_sim > MAX_REWARD:
                    trial_raw_score = 1.0
                trial_event_list.append(
                    (f"trial{trial_index}_feedback", f"{timestamp}:2")
                )
                block_feedback_event_list.append(f"{timestamp}*{trial_raw_score}:2")

                trial_index += 1

        trial_index -= 1
        if trial_index != 8:
            # Incomplete trial, this subject needs to be excluded
            print(f"Incomplete trial!: {subject_id} - {run_id}")
            mark_subject_exclusion(
                subject_id, f"Incomplte trials ({trial_index}) at {run_id}"
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

        # Store trial-wise regressors
        try:
            trial_event_order_list = [event[0] for event in trial_event_list]
            with open(
                run_regressors_dir
                / f"{subject_id}_task-photographer_{run_id}_trial_event_order.1D",
                "w",
            ) as f:
                f.writelines("\n".join(trial_event_order_list))
        except OSError:
            raise RuntimeError(
                f"Cannot write trial_event_order.1D for {subject_id}, {run_id}."
            )

        for trial_event_type, trial_event_annotation in trial_event_list:
            try:
                with open(
                    run_regressors_dir
                    / f"{subject_id}_task-photographer_{run_id}_trial_{trial_event_type}_event.1D",
                    "w",
                ) as f:
                    f.write(f"{trial_event_annotation}\n")
            except OSError:
                raise RuntimeError(
                    f"Cannot write trial_{trial_event_type}_event.1D for {subject_id}, {run_id}."
                )

        # Store block-wise regressors
        block_event_list_dict = {
            "exploration": block_exploration_event_list,
            "capture": block_capture_event_list,
            "capture_failed": block_capture_failed_event_list,
            "preview": block_preview_event_list,
            "voice": block_voice_event_list,
            "caption": block_caption_event_list,
            "feedback": block_feedback_event_list,
        }

        for block_event_type, block_event_list in block_event_list_dict.items():
            if len(block_event_list) > 0:
                try:
                    with open(
                        run_regressors_dir
                        / f"{subject_id}_task-photographer_{run_id}_block_{block_event_type}_event.1D",
                        "w",
                    ) as f:
                        f.write(" ".join(block_event_list) + "\n")

                except OSError:
                    RuntimeError(
                        f"Cannot write block_{block_event_type}_event.1D for {subject_id}, {run_id}."
                    )


def prepare_task_stim(config):
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
        _subject_task_stim(subject_id, config)

    # Delete excluded subjects
    delete_marked_subjects(config)
