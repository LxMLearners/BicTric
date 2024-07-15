# BicTric: Learning prognostic models using a mixture of biclustering and triclustering

## Introduction

## Getting Started

These instructions will get you how to run the BicTric classifier.

### Prerequisites

To run the BicTric classifier you need to have Python 3.4 or above installed as well as the following packages:

- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [scipy](https://scipy.org/install.html)
- [numpy](https://numpy.org/install/)
- [pandas](https://pandas.pydata.org/getting_started.html)
- [sortedcontainers](http://www.grantjenks.com/docs/sortedcontainers/#quickstart)

### Config File (YAML)

BicTric runs using parameters defined in a yaml config file.

Example:

```YAML
DATA_FILE: <path_to_data_file>
TOP_FOLDER: <path_to_outputs_top_folder>
TARGET: <TARGET class>
CS_STRATEGY: <"strict" OR "flexible">
DISCRETIZATION: False
PATTERNS_3D: False
STAGES: False
FEATURE_SELECTION: False
STATIC_FEATURES: {
                  'Gender':'categorical',
                  'Age_onset':'continuos',
                  'ALS_familiar_history':'categorical',
                  'UMNvsLMN':'categorical',
                  'Onset':'categorical',
                  'C9orf72':'categorical'
                }
TEMPORAL_FEATURES: {
                  'ALSFRSb':'categorical',
                  'ALSFRSsUL':'categorical',
                  'ALSFRSsLL':'categorical',
                  'R':'categorical',
                  'ALSFRS-R':'categorical',
                  '%FVC':'continuos',
                  'MITOS-stage':'categorical'
                  }

```

`CS_STRATEGY`: Strategy to be used in the process of creating the sets of snapshots. Flexible if we are using a minimum lenght or Strict if we want strict lenght of sets.

### How to run

1. Generate longitudinal tables

`$ python3 longitudinal_tables_strat.py <n> <config_file_name>`

2. Run BicTric pipeline

`$ python3 bictric.py <n> <config_file> [<cr_point>] [<tw>] [<group>]`

## Citing the Paper 📑

If you use the BicTric classifier in your research, please cite our paper:

_Soares, D. F., Henriques, R., Gromicho, M., de Carvalho, M., & C Madeira, S. (2022) Learning prognostic models using a mixture of biclustering and triclustering: Predicting the need for non-invasive ventilation in Amyotrophic Lateral Sclerosis. Journal of Biomedical Informatics, 134, 104172_ https://doi.org/10.1016/j.jbi.2022.104172
