import json
import shutil
from pathlib import Path

from .types import ConfigDict, ExclusionDict


def _check_subject_exclusion_config(config: ConfigDict):
    try:
        subject_exclusion_file_path = config["execution"]["subject_exclusion_file_path"]
        return Path(subject_exclusion_file_path)
    except KeyError:
        raise RuntimeError('"subject_exclusion_file_path" is not set in config.')


def _read_or_create_subject_exclusion(config: ConfigDict):
    subject_exclusion_file_path = _check_subject_exclusion_config(config)

    if subject_exclusion_file_path.exists():
        try:
            with open(subject_exclusion_file_path, "r") as f:
                return ExclusionDict(json.load(f))
        except IOError:
            raise RuntimeError(
                f'Cannot read "subject_exclusion.json": <{subject_exclusion_file_path}>'
            )
        except json.JSONDecodeError as json_error:
            raise RuntimeError(
                f'"subject_exclusion.json" seems corrupted or malformed: <{json_error.lineno}:{json_error.msg}>'
            )
    else:
        return ExclusionDict({})


def _write_subject_exclusion(exclusion_dict: ExclusionDict, config: ConfigDict):
    subject_exclusion_file_path = _check_subject_exclusion_config(config)

    try:
        with open(subject_exclusion_file_path, "w") as f:
            return json.dump(exclusion_dict, f, indent=2)
    except IOError:
        raise RuntimeError(
            f'Cannot write to "subject_exclusion.json": <{subject_exclusion_file_path}>'
        )


def mark_subject_exclusion(subject_id: str, reason: str, config: ConfigDict):
    exclusion_dict = _read_or_create_subject_exclusion(config)

    if subject_id in exclusion_dict:
        if reason not in exclusion_dict[subject_id]:
            exclusion_dict[subject_id].append(reason)

    else:
        exclusion_dict[subject_id] = [reason]

    _write_subject_exclusion(exclusion_dict, config)


def delete_marked_subjects(config: ConfigDict):
    exclusion_dict = _read_or_create_subject_exclusion(config)

    output_dir = Path(config["execution"]["output_dir"])
    assert output_dir.exists(), f"Output directory is not found: <{output_dir}>"

    for subject_id in exclusion_dict.keys():
        subject_output_dir = output_dir / subject_id
        shutil.rmtree(subject_output_dir, ignore_errors=True)


def read_subject_exclusion(config: ConfigDict):
    subject_exclusion_file_path = _check_subject_exclusion_config(config)

    try:
        with open(subject_exclusion_file_path, "r") as f:
            return ExclusionDict(json.load(f))
    except IOError:
        raise RuntimeError(
            f'Cannot read "subject_exclusion.json": <{subject_exclusion_file_path}>'
        )
    except json.JSONDecodeError as json_error:
        raise RuntimeError(
            f'"subject_exclusion.json" seems corrupted or malformed: <{json_error.lineno}:{json_error.msg}>'
        )
