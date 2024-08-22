import os
from pathlib import Path

import nibabel as nib
import numpy as np


class NiftiImage:
    def __init__(self, data, dim=None, affine=None):
        self.data = data
        self.dim = dim
        self.affine = affine


def load_nifti(
    path: Path, save_dim: None | bool = None, save_affine: None | bool = None
):
    try:
        nifti_image = nib.load(path)
        nifti_data = nifti_image.get_fdata()

        nifti_dim, nifti_affine = None, None

        if save_dim:
            nifti_dim = nifti_data.shape
        if save_affine:
            nifti_affine = nifti_image.affine

        return NiftiImage(nifti_data, nifti_dim, nifti_affine)

    except Exception as e:
        raise e


def save_nifti(data: np.ndarray, base: NiftiImage, path: Path, file_name: str):
    if base.affine is None or base.dim is None:
        raise ValueError('base image should have "affine" or "dim" attributes.')

    if data.shape != base.dim:
        raise ValueError("data.shape != base.dim.")

    new_nifti_image = nib.Nifti1Image(data, affine=base.affine)

    try:
        if not path.exists():
            os.makedirs(path, exist_ok=True)
    except OSError:
        raise RuntimeError(f"Cannot create a new directory at <{path}>.")

    try:
        nib.save(new_nifti_image, path / f"{file_name}.nii")
    except Exception as e:
        print(e)
        raise RuntimeError(f"Cannot save a NIFTI image ({file_name}) at <{path}>.")
