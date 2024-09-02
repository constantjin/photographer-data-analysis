# Photographer Paradigm Data Analysis Scripts

Python/R scripts for neural/behavioral data from the Photographer Paradigm

## Requirements

### Neural and behavioral data

The orignal neural and behavioral data acquired from the Photographer experiment through 2022 and 2023 are available on request from the corresponding author. Please contact us with respect to the BSPL "[Contact Us](https://bspl-ku.github.io/contact/)" page. 

### Dependencies

- Python 3.10.8
    - If you use [asdf](https://asdf-vm.com/) for toolchain management, you can run `asdf install` from the project root. 

    - We use [Poetry](https://python-poetry.org/) for dependency management.
        - Please install dependencies by running `poetry install` from the project root.

- R 4.2.2
    - Session Information

        ```R
        R version 4.2.2 (2022-10-31 ucrt)
        Platform: x86_64-w64-mingw32/x64 (64-bit)
        Running under: Windows 10 x64 (build 22631)

        Matrix products: default

        locale:
        [1] LC_COLLATE=Korean_Korea.utf8  LC_CTYPE=Korean_Korea.utf8    LC_MONETARY=Korean_Korea.utf8
        [4] LC_NUMERIC=C                  LC_TIME=Korean_Korea.utf8    

        attached base packages:
        [1] stats     graphics  grDevices utils     datasets  methods   base     

        other attached packages:
        [1] slider_0.3.0        glue_1.6.2          emmeans_1.8.5       AICcmodavg_2.3-2    broom.mixed_0.2.9.4
        [6] lmerTest_3.1-3      lme4_1.1-32         Matrix_1.5-1        ggpubr_0.6.0        rstatix_0.7.2      
        [11] ggnewscale_0.4.9    ggforce_0.4.1       see_0.8.0           report_0.5.7        parameters_0.21.1  
        [16] performance_0.10.4  modelbased_0.8.6    insight_0.19.7      effectsize_0.8.5    datawizard_0.8.0   
        [21] correlation_0.8.4   bayestestR_0.13.1   easystats_0.6.0     lubridate_1.9.2     forcats_1.0.0      
        [26] stringr_1.5.0       dplyr_1.1.1         purrr_1.0.1         readr_2.1.4         tidyr_1.3.0        
        [31] tibble_3.2.0        ggplot2_3.5.1       tidyverse_2.0.0    
        ```

## First-level analysis

### Goal

Run block-/trial-wise GLM, perform searchlight representational similarity analysis (RSA), and conduct one-sample t-tests on individual RSA maps on each subgroup (either Discovery or Validation group)

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