import similarity_utils as su
import constants
from tricluster import Tricluster
import pandas as pd


def translate_features(s):
    d = {
        'S-0': 'ALS-FRS',
        'S-1': 'ALS-FRS-R',
        'S-2': 'ALS-FRSb',
        'S-3': 'ALS-FRSsUL',
        'S-4': 'ALS-FRSsLL',
        'S-5': 'ALS-FRSr',
        'S-6': 'R',
        'S-7': 'FVC',
        'S-8': 'MIP',
        'S-9': 'MEP',
        'S-10': 'prog_rate',
        'S-11': 'mitos_s'
    }
    return d[s]


def translate_features_2d(s):
    d = {
        'S-0': 'Gender',
        'S-1': 'BMI',
        'S-2': 'MND familiar history',
        'S-3': 'Age at onset',
        'S-4': 'Disease duration',
        'S-5': 'El Escorial reviewed criteria',
        'S-6': 'UMN vs LMN',
        'S-7': 'Onset form',
        'S-8': 'C9orf72'
    }
    return d[s]


class Patterns:

    def load_triclusters(self, filename):
        self._triclusters = su.get_triclusters(filename)

    def get_pattern(self, tric_n):
        f_categorical_ix = list(map(lambda f: f"S-{list(constants.TEMPORAL_FEATURES.keys()).index(f[0])}", filter(
            lambda x: x[1] == 'categorical', constants.TEMPORAL_FEATURES.items())))
        f_continuos_ix = list(map(lambda f: f"S-{list(constants.TEMPORAL_FEATURES.keys()).index(f[0])}", filter(
            lambda x: x[1] == 'continuos', constants.TEMPORAL_FEATURES.items())))
        return su.compute_representative_patterns(self._triclusters[tric_n], f_categorical_ix, f_continuos_ix)

    def print_pattern(self, tric_n):
        rp = self.get_pattern(tric_n)
        ret = ""
        i = 0
        for p in rp:
            ret += "Tric_{}_{}: [".format(tric_n, i)
            for f in p:
                f_name = list(constants.TEMPORAL_FEATURES.keys())[
                    int(f[0].split("-")[1])]
                ret += "{}={}".format(f_name, f[1])
                if f != p[-1]:
                    ret += ', '
            ret += "]\n"
            i += 1
        return ret

    def print_pattern_i(self, tric_n, t_ix):
        p = self.get_pattern(tric_n)[t_ix]
        ret = ""
        ret += "Tric_{}_{}: [".format(tric_n, t_ix)
        for f in p:
            f_name = list(constants.TEMPORAL_FEATURES.keys())[
                int(f[0].split("-")[1])]
            ret += "{}={}".format(f_name, f[1])
            if f != p[-1]:
                ret += ', '
        ret += "]\n"

        return ret

    def patient_triclusters(self, n, baseline_file):
        mats = pd.read_csv(baseline_file)
        mats = mats.fillna(0)
        mats.drop(columns=[constants.TARGET], inplace=True)
        X_res = mats.loc[:, ].values

        n_feats = int((len(mats.columns)-1)/n)

        ps_tr = list()
        for e in X_res:
            p_tric = Tricluster(n, n_feats, 1)
            i = 0
            f = 0

            for v in e[1:]:
                p_tric.addValue("T-"+str(i), "S-"+str(f), "G-" + str(e[0]), v)
                if f == (n_feats-1):
                    f = 0
                    i += 1
                else:
                    f += 1
            ps_tr.append(p_tric)
        self._triclusters = ps_tr


class Patterns2D:

    def load_biclusters(self, filename):
        self._biclusters = su.get_triclusters(filename)

    def get_pattern(self, tric_n):
        f_cat_static = list(map(lambda f: f"S-{list(constants.STATIC_FEATURES.keys()).index(f[0])}", filter(
            lambda x: x[1] == 'categorical', constants.STATIC_FEATURES.items())))
        f_cont_static = list(map(lambda f: f"S-{list(constants.STATIC_FEATURES.keys()).index(f[0])}", filter(
            lambda x: x[1] == 'continuos', constants.STATIC_FEATURES.items())))
        return su.compute_representative_patterns(self._biclusters[tric_n], f_cat_static, f_cont_static)

    def print_pattern(self, tric_n):
        rp = self.get_pattern(tric_n)
        ret = ""
        i = 0
        for p in rp:
            ret += "Bic_{}_{}: [".format(tric_n, i)
            for f in p:
                f_name = list(constants.STATIC_FEATURES.keys())[
                    int(f[0].split("-")[1])]
                ret += "{}={}".format(f_name, f[1])
                if f != p[-1]:
                    ret += ', '
            ret += "]\n"
            i += 1
        return ret

    def print_pattern_i(self, tric_n, t_ix):
        p = self.get_pattern(tric_n)[t_ix]
        ret = ""
        ret += "Bic_{}_{}: [".format(tric_n, t_ix)
        for f in p:
            f_name = list(constants.STATIC_FEATURES.keys())[
                int(f[0].split("-")[1])]
            ret += "{}={}".format(f_name, f[1])
            if f != p[-1]:
                ret += ', '
        ret += "]\n"

        return ret

    def patient_biclusters(self, n, baseline_file):
        mats = pd.read_csv(baseline_file)
        mats.fillna('0', inplace=True)
        # mats = label_encoder_als(mats, constants.STATIC_FEATURES)
        mats = mats.round(1)

        mats.drop(columns=[constants.TARGET], inplace=True)
        X_res = mats.loc[:, ].values

        ps_tr = list()
        feats = constants.STATIC_FEATURES
        pro_rates_res = list()
        for e in X_res:
            p_tric = Tricluster(n, len(feats), 1)
            i = 0
            f = 0
            for v in e[1:]:
                p_tric.addValue("T-"+str(i), "S-"+str(f),
                                "G-" + str(e[0]), float(v))
                if f == len(feats)-1:
                    f = 0
                    i += 1
                else:
                    f += 1
            pro_rates_res.append(e[-1])
            ps_tr.append(p_tric)

        self._biclusters = ps_tr
