import gc
import os
import subprocess
import time
from pathlib import Path

VOXELWISE_P_THRESHOLD = 0.005
VOXELWISE_Z_THRESHOLD = 2.5758
CLUSTER_LEVEL_ALPHA = 0.05


def extract_feedback_rsa_cluster_mask(config):
    output_dir = Path(config["execution"]["output_dir"])
    assert output_dir.exists(), f"Output directory is not found: <{output_dir}>"

    rsa_feedback_model_name_list = [
        "current_trial",
        "one_back_trial",
        "two_back_trial",
        "recent_2_trial",
        "recent_3_trial",
        "previous_2_trial",
    ]

    searchlight_radius = config["execution"]["rsa"]["searchlight_radius"]
    blur_kernel_width = config["execution"]["rsa"]["rsa_blur_kernel_width"]

    for rsa_feedback_model_name in rsa_feedback_model_name_list:
        gc.collect()

        rsa_model_ttest_dir = (
            output_dir
            / "stat"
            / "multivariate"
            / "feedback_model"
            / "ttest"
            / rsa_feedback_model_name
        )

        if not rsa_model_ttest_dir.exists():
            raise RuntimeError(
                f'RSA map t-test result directory not found: <{rsa_model_ttest_dir}>. Please run "stat.run_feedback_rsa_ttest" task first.'
            )

        os.chdir(rsa_model_ttest_dir)

        rsa_stat_map_name = f"feedback_rsa_ttest_{rsa_feedback_model_name}_within_run_mean_rad{searchlight_radius}_blur{blur_kernel_width}.nii"

        print(f"Extract cluster mask from the {rsa_feedback_model_name} stat map file:")

        # Check presence of the RSA statistical map
        rsa_stat_map_path = rsa_model_ttest_dir / rsa_stat_map_name
        assert rsa_stat_map_path.exists()

        # Read the 3dClustsim table and get the minimal cluster extent
        rsa_clustersim_table_file_name = f"feedback_rsa_ttest_{rsa_feedback_model_name}_within_run_mean_rad{searchlight_radius}_blur{blur_kernel_width}.CSimA.NN2_1sided.1D"
        try:
            minimum_cluster_extent = int(
                subprocess.check_output(
                    f"1d_tool.py -infile {rsa_clustersim_table_file_name} -csim_show_clustsize -csim_pthr {VOXELWISE_P_THRESHOLD} -csim_alpha {CLUSTER_LEVEL_ALPHA} -verb 0",
                    shell=True,
                    text=True,
                )
            )
            print(
                f"# Minimum cluster extent (voxel-wise P < {VOXELWISE_P_THRESHOLD}, cluster-level alpha < {CLUSTER_LEVEL_ALPHA}): {minimum_cluster_extent} voxels"
            )
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"Cannot extract cluster extent from {rsa_clustersim_table_file_name} using 1d_tool.py."
            )

        # Extract the clusters using 3dClusterize
        try:
            thresholded_cluster_map_name = f"feedback_rsa_ttest_{rsa_feedback_model_name}_rad{searchlight_radius}_blur{blur_kernel_width}_clusters"

            Path(
                rsa_model_ttest_dir / f"{thresholded_cluster_map_name}+tlrc.HEAD"
            ).unlink(missing_ok=True)
            Path(
                rsa_model_ttest_dir / f"{thresholded_cluster_map_name}+tlrc.BRIK.gz"
            ).unlink(missing_ok=True)

            subprocess.run(
                f"3dClusterize -nosum -1Dformat -inset {rsa_stat_map_name} -idat 1 -ithr 1 -NN 2 -clust_nvox {minimum_cluster_extent} -1sided RIGHT_TAIL {VOXELWISE_Z_THRESHOLD} -mask mni_152_gm_mask_3mm.nii -pref_map {thresholded_cluster_map_name}",
                shell=True,
            )
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"Cannot extract clusters from {rsa_stat_map_name} using 3dClusterize."
            )

        # Gather all clusters into a binary mask using 3dcalc
        try:
            final_cluster_mask_name = f"feedback_rsa_ttest_{rsa_feedback_model_name}_rad{searchlight_radius}_blur{blur_kernel_width}_cluster_mask.nii"
            subprocess.run(
                f"3dcalc -a {thresholded_cluster_map_name}+tlrc -expr 'step(a)' -prefix {final_cluster_mask_name} -overwrite",
                shell=True,
            )
        except Exception as e:
            print(e)
            raise RuntimeError("Cannot create the cluster mask using 3dcalc.")

        print()
        time.sleep(2)
