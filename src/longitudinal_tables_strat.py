import preprocessing.als_preprocess as als
import preprocessing.als_imbl as alsi

import pandas as pd
import constants
import sys
from pathlib import Path


def load_data_baselines(features, group):
    infile = constants.DATA_FILE.format(group)

    data = pd.read_csv(infile)

    data['Evolution'] = 'No'

    data = data[features]

    print("y INI : ", data['Evolution'].value_counts())

    return data


def compute_temporal_table(data, n, features):

    data = data[features]
    data_dict = als.df_to_dict(data)

    sps = als.compute_consecutive_snapshots_n(
        data_dict, n, 'Evolution', strategy=constants.CS_STRATEGY)

    mats, y = als.create_matrix_temporal(data_dict, sps, n)

    # mats.fillna(0, inplace=True)

    baseline_temporal = constants.BASELINE_DIR_T + f"{n}TPS/"

    Path(baseline_temporal).mkdir(parents=True, exist_ok=True)
    mats['Evolution'] = 'No'

    mats.to_csv(baseline_temporal +
                f"{n}TPS_baseline_temporal.csv", index=False)


def compute_static_table(data, n, features):
    data = data[features]

    data_dict = als.df_to_dict(data)

    sps = als.compute_consecutive_snapshots_n(
        data_dict, n, 'Evolution', strategy=constants.CS_STRATEGY)

    mats, y = als.create_matrix_static(data_dict, sps)

    mats = alsi.label_encoder_als(
        mats, ['Gender', 'UMNvsLMN', 'C9orf72'])

    mats.fillna(0, inplace=True)

    baseline_static = constants.BASELINE_DIR_S + "{n}TPS/"

    Path(baseline_static).mkdir(parents=True, exist_ok=True)
    mats['Evolution'] = 'No'

    mats.to_csv(baseline_static+f"{n}TPS_baseline_static.csv", index=False)


n = int(sys.argv[1])
constants.get_config(sys.argv[2])
group = ""

features = ['REF'] + list(constants.STATIC_FEATURES.keys()) + \
    list(constants.TEMPORAL_FEATURES.keys()) + ['Evolution']

data = load_data_baselines(features, group)

t = compute_temporal_table(
    data, n, ['REF'] + list(constants.TEMPORAL_FEATURES.keys()) + ['Evolution'])


s = compute_static_table(data, n, ['REF'] +
                         list(constants.STATIC_FEATURES.keys()) + ['Evolution'])
