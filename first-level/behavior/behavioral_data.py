from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from ..utils.path import get_fmriprep_output_dir
from ..utils.subject_exclusion import read_subject_exclusion
from ..utils.types import ConfigDict

MAX_REWARD = 0.28
YOLO_MODEL_NAME = "yolov5s"


def _gather_subject_feedback_data(subject_id: str, yolo_model: any, config: ConfigDict):
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

    print(subject_id)

    # Find run_id - run directory pairs
    rsa_city_list = ["New_York", "Boston", "Los_Angeles", "London", "Paris"]
    run_id_city_name_city_dir_pair_list = []  # list of (run_id, run_dir)
    for city_name in rsa_city_list:
        run_city_dir = [
            child for child in subject_behavior_dir.iterdir() if city_name in child.stem
        ][0]
        run_id = f'run-0{run_city_dir.stem.split("_")[0]}'
        run_id_city_name_city_dir_pair_list.append((run_id, city_name, run_city_dir))
    run_id_city_name_city_dir_pair_list.sort(
        key=lambda pair: int(pair[0].split("-")[1])
    )

    assert (
        [pair[0] for pair in run_id_city_name_city_dir_pair_list]
        == [
            "run-01",
            "run-02",
            "run-03",
            "run-04",
            "run-05",
        ]
    ), f"Run_id - run_city_dir pair list is not sorted: <{run_id_city_name_city_dir_pair_list}>"

    subject_feedback_dict_list = []

    for run_id, city_name, run_city_dir in run_id_city_name_city_dir_pair_list:
        run_etime_path = Path(run_city_dir) / "log_etime.txt"

        trial_index = 1

        if not run_etime_path.exists():
            raise RuntimeError(
                f"Cannot find etime file for {subject_id} {run_id}: <{run_etime_path}>"
            )

        try:
            with open(run_etime_path, "r") as f:
                run_etime_lines = f.readlines()
        except IOError:
            raise RuntimeError(
                f"Cannot read etime file for {subject_id} {run_id}: <{run_etime_path}>"
            )

        for etime_line in run_etime_lines:
            trial_line = etime_line.strip().split("\t")[
                -1
            ]  # Exclude timestamp information

            if "trial_reward" in trial_line:
                current_score = float(trial_line.split("/percent:")[-1])
                current_sim = float(trial_line.split("/")[0].split(":")[-1])

                if current_sim > MAX_REWARD:
                    current_score = 100  # correcting score (percent) errors

                # last event in a run -> perform YOLOv5 inference
                trial_image_path = run_city_dir / "capture" / f"trial_{trial_index}.png"

                pil_image = Image.open(trial_image_path)
                results = yolo_model(pil_image)
                categories = results.pandas().xyxy[0]["name"].tolist()

                person_present = 1 if "person" in categories else 0
                bicycle_present = 1 if "bicycle" in categories else 0
                traffic_present = 1 if "traffic light" in categories else 0

                subject_feedback_dict_list.append(
                    {
                        "subject_id": subject_id,
                        "city": city_name,
                        "run": int(run_id.split("-")[-1]),
                        "trial": trial_index,
                        "feedback_score": current_score,
                        "cosine_similarity": current_sim,
                        "person": person_present,
                        "bicycle": bicycle_present,
                        "traffic_light": traffic_present,
                    }
                )

                trial_index += 1

    subject_feedback_df = pd.DataFrame.from_records(subject_feedback_dict_list)
    return subject_feedback_df


def prepare_behavioral_data(config: ConfigDict):
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

    all_subject_feedback_df_list = []

    behavioral_data_dir_path = Path(config["execution"]["glm"]["behavioral_data_dir"])
    subgroup = str(config["execution"]["bids_dir"]).split("/bids")[0].split("/")[-1]
    print(f"Subgroup: {subgroup}")

    # Prepare YOLOv5 model
    yolo_model = torch.hub.load("ultralytics/yolov5", YOLO_MODEL_NAME)

    for subject_id in subject_list:
        df = _gather_subject_feedback_data(subject_id, yolo_model, config)
        all_subject_feedback_df_list.append(df)

    subject_feedback_df: pd.DataFrame = pd.concat(
        all_subject_feedback_df_list, ignore_index=True
    )

    subject_feedback_df.to_csv(
        behavioral_data_dir_path / f"group_{subgroup}_behavior_feedback.csv",
        index=False,
    )
