from pathlib import Path

from .types import ConfigDict


def get_fmriprep_output_dir(config: ConfigDict):
    fmriprep_output_dir = Path(
        config["execution"]["bids_dir"]
        / "derivatives"
        / f'fmriprep-{config["execution"]["glm"]["fmriprep_version"]}-{"reconall" if config["execution"]["glm"]["performed_reconall"] else ""}'
    )

    if fmriprep_output_dir.exists():
        return fmriprep_output_dir
    else:
        raise RuntimeError(
            f"fMRIPrep output directory is not found: <{fmriprep_output_dir}>"
        )
