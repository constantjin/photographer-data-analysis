# Photographer Paradigm Data Analysis Scripts

Python/R scripts for neural/behavioral data from the [Photographer Paradigm](https://github.com/constantjin/photographer-experimental-paradigm)

## References

  - **Jin, S.**, Lee, J., & Lee, J. H. Historical Feedback Representations Robustly Guide Learning. Organization for Human Brain Mapping (OHBM) 2024. Seoul, Korea.
  - **Jin, S.**, Lee, J., & Lee, J. H. How to Be a Good Photographer: Multi-modal Learning In a Real-life Environment. OHBM 2023. Montreal, Canada. **[Oral Presentation]**

## Requirements

### Neural and behavioral data

The orignal neural and behavioral data acquired from the Photographer experiment through 2022 and 2023 are available on request from the corresponding author. Please contact us with respect to the BSPL "[Contact Us](https://bspl-ku.github.io/contact/)" page. 

- **Update**
  - Preprocessed behavioral data, cross-validated cluster mask files, and second-level RSA results for the second-level analysis or R visualization scripts would be open to public after our manuscript is submitted.

### Dependencies

- Python 3.10.8
    - If you use [asdf](https://asdf-vm.com/) for toolchain management, you can run `asdf install` from the project root. 

    - We use [Poetry](https://python-poetry.org/) for dependency management.
        - Please install dependencies by running `poetry install` from the project root.

- R >= 4.2.2
    - Session Information

        ```R
        R version 4.4.1 (2024-06-14 ucrt)
        Platform: x86_64-w64-mingw32/x64
        Running under: Windows 11 x64 (build 22631)

        Matrix products: default


        locale:
        [1] LC_COLLATE=Korean_Korea.utf8  LC_CTYPE=Korean_Korea.utf8    LC_MONETARY=Korean_Korea.utf8 LC_NUMERIC=C                 
        [5] LC_TIME=Korean_Korea.utf8    

        time zone: Asia/Seoul
        tzcode source: internal

        attached base packages:
        [1] stats     graphics  grDevices utils     datasets  methods   base     

        other attached packages:
        [1] slider_0.3.1        glue_1.7.0          emmeans_1.10.4      AICcmodavg_2.3-3    broom.mixed_0.2.9.5 lmerTest_3.1-3     
        [7] lme4_1.1-35.5       Matrix_1.7-0        ggpubr_0.6.0        rstatix_0.7.2       ggnewscale_0.5.0    ggforce_0.4.2      
        [13] see_0.9.0           report_0.5.9        parameters_0.22.2   performance_0.12.3  modelbased_0.8.8    insight_0.20.4     
        [19] effectsize_0.8.9    datawizard_0.12.3   correlation_0.8.5   bayestestR_0.14.0   easystats_0.7.3     lubridate_1.9.3    
        [25] forcats_1.0.0       stringr_1.5.1       dplyr_1.1.4         purrr_1.0.2         readr_2.1.5         tidyr_1.3.1        
        [31] tibble_3.2.1        ggplot2_3.5.1       tidyverse_2.0.0    
        ```

- [AFNI](https://afni.nimh.nih.gov/) for neuroimaging analysis

## First-level analysis

### Goal

- Run block-/trial-wise GLM
- Perform searchlight representational similarity analysis (RSA)
- Conduct one-sample t-tests on individual RSA maps

### Prerequisites

- Preprocessed fMRI data via fMRIPRep
    - Note: We used the following docker command to preprocess the BIDS dataset:

        ```bash
        docker run -it --rm -v ${PWD}:/data nipreps/fmriprep:23.0.2 /data/bids /data/bids/derivatives/fmriprep-23.0.2-reconall participant -w /data/tmp/workdir --skip-bids-validation --fs-license-file /data/bids/derivatives/license.txt --output-spaces MNI152NLin2009cAsym:res-2 MNI152NLin2009cAsym --stop-on-first-crash
        ```

- `photographer_config.toml` in the `[BIDS root]/bids/code` directory 
    - We manage analysis parameters/choices in a separate toml file.
    - Please refer to our `photographer_config.toml` file in the BIDS data directories.
    - You can find documentations for keys in the config file from the [`first-level/utils/types.py`](first-level/utils/types.py).

- MNI152NLin2009cAsym gray matter (GM) templete
    - You can download the template file from [this link](http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09c_nifti.zip).
    - Decompress the ZIP file and you can find the `mni_icbm152_gm_tal_nlin_asym_09c.nii` file.
    - Please update the `mask.mni_gm_template_path` key from the `photographer_config.toml` file accordingly.

### Command

Since the `first-level` package is designed as a BIDS-App, you can run first-level tasks using the following command:

```
usage: python -m first-level [-h] [--participant-label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]] 
                             [--config-file CONFIG_FILE]
                             -t TASK_NAME
                             bids_dir output_dir {participant}

Photographer Data First-level Analysis

Positional arguments:
  bids_dir              The BIDS root directory.
  output_dir            The analysis output directory. It should be (bids_dir)/derivatives/first-level
  {participant}         Processing stage. Only "participant" is accepted. I believe this is in the BIDS-Apps spec?

Options:
  -h, --help            show this help message and exit

BIDS-related argument:
  --participant-label, --participant_label
                        A single participant label or a space-separated participant labels.

First-level analysis-related arguments:
  -t, --task TASK_NAME
                        A first-level analysis task to run.
  --config-file, --config_file CONFIG_FILE
                        A config file (toml) path. If not specified, we will try to find photographer_config.toml in (bids_dir)/code.
```

### Tasks

> Note that the `Order` column represents recommended task orders.

| Order | Task Name | Description |
| ----- | --------- | ----------- |
| 1 | `glm.prepare_task_stim` | Prepare task-related GLM regressors from behavioral data. |
| 2 | `glm.prepare_confound` | Prepare nuisance GLM regressors from fMRIPrep confounds. |
| 3 | `glm.run_block_wise_glm` | Run block-wise GLM (GLM1) for univariate analysis. |
| 4 | `glm.run_trial_wise_glm` | Run trial-wise GLM (GLM2) for multivariate (RSA) analysis. |
| 5 | `mask.prepare_gm_mask` | Prepare a gray matter (GM) mask from the MNI152NLin2009cAsym GM template. |
| 6 | `behavior.prepare_behavioral_data` | Preprocess behavioral data into a CSV file and include object detection results. |
| 7 | `rsa.prepare_feedback_neural_data` | Aggregate trial-wise feedback event beta maps from GLM 2 into a numpy array (NPY) file |
| 8 | `rsa.prepare_feedback_model_rdm` | Prepare feedback history model RDMs from the preprocessed behavioral data. |
| 9 | `rsa.run_feedback_rsa` | Run searchlight RSA on feedback event beta maps and feedback history model RDMs. |
| 10 | `stat.run_univariate_ttest` | Conduct t-tests on individual beta maps from GLM 1 (univariate analysis) |
| 11 | `stat.run_feedback_rsa_ttest` | Conduct t-tests on individual feedback history RSA maps. |
| 12 | `stat.extract_feedback_rsa_cluster_mask` | Compute corrected cluster masks from feedback history RSA statistical maps. |

### Acknowledgments

- [fMRIPrep GitHub](https://github.com/nipreps/fmriprep) for the BIDS-App structure reference.
- [Minseok Choi](https://github.com/BigJade-C) originally implemented codes for computing searchlight spheres (`first-level/utils/searchlight.py`).

## Second-level analysis

### Goal

- Identify cross-validated RSA clusters across both subgroups (i.e., Discovery and Validation group)
- Perform second-level RSA (using both feedback history and exploration models) on the robust RSA clusters (i.e., MiOG and IFG) representing feedback history information

### Scripts (IPython Notebook)

> Please refer to notes and comments in each .ipynb script for detailed information.

- [`1_cross_validated_rsa_clusters.ipynb`](second-level/1_cross_validated_rsa_clusters.ipynb): Compute cross-validated cluster masks; Identify Feedback History clusters (i.e., MiOG and IFG) from the cross-validated cluster masks

- [`2_visualization_of_rsa_clusters.ipynb`](second-level/2_visualization_of_rsa_clusters.ipynb): Plot surface-based mapping for the cross-validated RSA clusters

- [`3_run_second_level_rsa.ipynb`](second-level/3_run_second_level_rsa.ipynb): Run second-level RSA on the Feedback History clusters using feedback history or exploration model RDMs

### Data and outputs

- Data
  - `photographer_exploration_info.csv`
    - For exploration model RDMs, we extracted exploration times, exploration distances, and coordinates for the capture locations from the etime files. This file is needed for the second-level RSA.

- Outputs
  - Cross-validated (feedback history) RSA cluster masks
    - For the Recent-2 Trial, Recent-3 Trial, and Feedback History (Recent-2 Trial & Recent-3 Trial) models
    - Will be stored in `second-level/output/cross_validated_cluster_mask`
    - We additionally identified two main RSA clusters, the MiOG and IFG, representing both Recent-2 Trial and Recent-3 Trial models.

  - CSV files for second-level RSA results
    - Note: Second-level RSA are restricted in the Feedback History clusters (i.e., MiOG and IFG regions).
    - Paths for storing RSA results
      - RSA using exploration models: `second-level/output/second_level_rsa/exploration_second_level`
      - Partial correlation RSA using feedback history models: `second-level/output/second_level_rsa/partial_feedback_history_second_level`
      - Partial correlation RSA using Capture Distance model: `second-level/output/second_level_rsa/partial_feedback_history_second_level`

### Acknowledgments

- [`surfplot`](https://github.com/danjgale/surfplot) for surface-based mapping
  - Gale, Daniel J., Vos de Wael., Reinder, Benkarim, Oualid, & Bernhardt, Boris. (2021). Surfplot: Publication-ready brain surface figures (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.5567926
  - Vos de Wael R, Benkarim O, Paquola C, Lariviere S, Royer J, Tavakol S, Xu T, Hong S-J, Langs G, Valk S, Misic B, Milham M, Margulies D, Smallwood J, Bernhardt BC. 2020. BrainSpace: a toolbox for the analysis of macroscale gradients in neuroimaging and connectomics datasets. *Communications Biology*. 3:103. https://doi.org/10.1038/s42003-020-0794-7

- [`neuromaps`](https://github.com/netneurolab/neuromaps) for volume-to-surface transformation (i.e., MNI152 to fsLR)
  - Buckner, R. L., Krienen, F. M., Castellanos, A., Diaz, J. C., & Yeo, B. T. (2011). The organization of the human cerebellum estimated by intrinsic functional connectivity. *Journal of Neurophysiology*, 106(5), 2322-2345. https://doi.org/10.1152/jn.00339.2011
  - Wu, J., Ngo, G. H., Greve, D., Li, J., He, T., Fischl, B., Eickhoff, S. B., & Yeo, B. T. T. (2018). Accurate nonlinear mapping between MNI volumetric and FreeSurfer surface coordinate systems. *Human Brain Mapping*  39(9), 3793-3808. https://doi.org/10.1002/hbm.24213


## R visualization scripts

### Goal

- Visualize behavioral data from the Photographer paradigm and perform validation of feedback and validation of learning
- Visualize second-level RSA results with respect to the clusters (MiOG or IFG) and subgroups (Discovery or Validation)

### Scripts (R Notebook)

> Please refer to notes and comments in each notebook for detailed information.

- [`1_photographer_behavioral_analysis.Rmd`](R_visualization/1_photographer_behavioral_analysis.Rmd) / [`.md`](R_visualization/1_photographer_behavioral_analysis.md): Perform statistical tests and visualize the behavioral data

- [`2_photographer_second_level_RSA_visualization.Rmd`](R_visualization/2_photographer_second_level_RSA_visualization.Rmd) / [`.md`](R_visualization/2_photographer_second_level_RSA_visualization.md): Visualize second-level RSA results (i.e., individual and mean RSA correlations) for each RSA cluster and subgroup

### Data and outputs

- Data
  - `group_*_behavior_feedback.csv`
    - Preprocessed behavioral data files from the `behavior.prepare_behavioral_data` task of the first-level analysis.
  
  - Second-level RSA CSV files
    - The same data copied from the second-level analysis (see the [Data and outputs](#data-and-outputs) section in the second-level analysis).

- Outputs
  - PNG files (300 dpi) representing figures. Most of them are separated by the subgroup name and/or RSA cluster name.