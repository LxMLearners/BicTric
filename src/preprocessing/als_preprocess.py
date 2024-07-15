import pandas as pd
import datetime as dt
import math
import numpy as np


def discretize(data_df, feats, q=[0, 0.25, 0.42, 0.58, 0.75, 1]):
    """
    Discretize Continuos Features in ALS data
    """

    for f in feats:
        data_df[f] = pd.qcut(data_df[f], q, range(1, len(q)))
    return data_df


def df_to_dict(data, discretize_prog_rate=False):
    """
    Transforms a DataFrame with ALS data into a Dict
    """
    # static progression rate
    def prog_rate(row):
        # print(row["1st symptoms"])
        # print(row["Date1"])
        if not (pd.isna(row["1st symptoms"]) or pd.isna(row["Date1"])):
            start_date = dt.datetime.strptime(row["1st symptoms"], '%d/%m/%Y')
            end_date = dt.datetime.strptime(row["Date1"], '%d/%m/%Y')
            num_months = (end_date.year - start_date.year) * \
                12 + (end_date.month - start_date.month)
            return (48 - row["ALS-FRS-R1"]) / num_months

    # temporal progression rate
    def prog_rate_now(row):
        # print(row["1st symptoms"])
        # print(row["Date1"])

        if not (pd.isna(row["1st symptoms"]) or pd.isna(row["medianDate"]) or pd.isna(row["ALS-FRS-R"])):
            start_date = dt.datetime.strptime(row["1st symptoms"], '%d/%m/%Y')
            end_date = dt.datetime.strptime(row["medianDate"], '%d/%m/%Y')
            num_months = (end_date.year - start_date.year) * \
                12 + (end_date.month - start_date.month)
            try:
                pr = (48 - row["ALS-FRS-R"]) / num_months
            except ZeroDivisionError:
                start_date = dt.datetime.strptime(
                    row["1st symptoms"], '%d/%m/%Y')
                end_date = dt.datetime.strptime(row["Date1"], '%d/%m/%Y')
                num_months = (end_date.year - start_date.year) * \
                    12 + (end_date.month - start_date.month)
                try:
                    pr = (48 - row["ALS-FRS-R1"]) / num_months
                except ZeroDivisionError:
                    pr = (48 - row["ALS-FRS-R1"]) / 1
            return pr
    try:
        data["prog_rate"] = data.apply(lambda row: prog_rate_now(row), axis=1)
        if discretize_prog_rate:
            data["prog_rate"] = pd.qcut(
                data["prog_rate"], [0, 0.25, 0.42, 0.58, 0.75, 1], range(1, 6))
    except:
        pass

    data_dict = data.to_dict('index')
    final_dict = dict()

    id_paciente_glob = 0
    time_counter = 0
    for k in data_dict.keys():
        ref = data_dict[k]['REF']  # ID Paciente da Iteração
        if ref != id_paciente_glob:
            id_paciente_glob = ref
            time_counter = 0

        del data_dict[k]['REF']
        if ref not in final_dict:
            final_dict[ref] = {time_counter: data_dict[k]}
        else:
            final_dict[ref][time_counter] = data_dict[k]

        time_counter += 1

    return final_dict


def replace_missings_by_zero(data):
    """
    Parameters
    ----------
    data: is a dict with ALS data with the format returned by `df_to_dict`
    """
    for (p, tps) in data.items():
        for (tp, fe) in tps.items():
            for f in fe.keys():
                try:
                    if math.isnan(data[p][tp][f]):
                        data[p][tp][f] = 0
                except:
                    pass

    return data


def scores_imputation(data):
    """
    Computes the ALSFRS scores which are missing

    Parameters
    -----------
    data: is a  dict {p_ID: {tp: {feat: val}}} as the output of `df_to_dict`
    """
    for (p, tps) in data.items():
        for tp in tps.keys():
            data[p][tp]['ALS-FRS'] = sum([data[p][tp]['P'+str(k)]
                                         for k in range(1, 11)])
            data[p][tp]['ALS-FRS-R'] = sum([data[p][tp]['P'+str(k)] for k in range(
                1, 10)]) + sum([data[p][tp][str(k)+'R'] for k in range(1, 4)])
            data[p][tp]['ALS-FRSb'] = sum([data[p][tp]['P'+str(k)]
                                          for k in range(1, 4)])
            data[p][tp]['ALS-FRSsUL'] = sum([data[p][tp]['P'+str(k)]
                                            for k in range(4, 7)])
            data[p][tp]['ALS-FRSsLL'] = sum([data[p][tp]['P'+str(k)]
                                            for k in range(7, 10)])
            data[p][tp]['ALS-FRSr'] = data[p][tp]['P10']
            data[p][tp]['R'] = sum([data[p][tp][str(k)+'R']
                                   for k in range(1, 4)])

    return data


def compute_consecutive_snapshots_n(data, n, label, yes_label='Y', strategy='flexible'):
    """

    Parameters
    ----------
    data: is a dict with ALS data with the format returned by `df_to_dict`
    n: is the number of consecutive snapshots to consider, ie. the size of snapshots set
        the size of snapshots set could be defined 
    label: is the target problem
    strategy: (default) `flexible` - sets of snapshots have a maximum size `n`
                `strict` - sets of snapshots have a strict size of `n`

    """

    final = dict()
    for (p, t) in data.items():
        if len(t.keys()) >= n:
            fd = dict()
            for (key, val) in t.items():
                fd[key] = val
                final[p] = fd

    snaps = dict()
    for (p, ts) in data.items():
        for t in ts.keys():

            size_t = len(ts.keys())
            if strategy == 'flexible':
                size_n = min(n, size_t)
            elif strategy == 'strict':
                size_n = n
            else:
                raise ValueError('Invalid Strategy')
            if t < size_t - (size_n-1) and all(map(lambda c: c != yes_label, [data[p][t+y][label] for y in range(0, size_n-1)])):
                if p not in snaps:
                    snaps[p] = list()
                snaps[p].append([(t+j, data[p][t+j][label])
                                for j in range(0, size_n)])
    return snaps


def remove_snapshots_all_null(data, snapshots):
    f = dict()
    for p in data.keys():
        t = data[p]
        remove = list()
        for tp in range(len(t.keys())):
            s = pd.Series([t[tp]['P1'], t[tp]['P2'], t[tp]['P3'], t[tp]['P4'], t[tp]['P5'], t[tp]
                          ['P6'], t[tp]['P7'], t[tp]['P8'], t[tp]['P9'], t[tp]['1R'], t[tp]['2R'], t[tp]['3R']])
            if s.isnull().values.all():
                remove.append(tp)
        f[p] = remove

    novos_snapshots = dict()
    for p in snapshots.keys():
        novo_l = list()
        for l in snapshots[p]:
            ls = list(map(lambda x: x[0], l))
            if len(set(ls) & set(f[p])) < 2:
                novo_l.append(l)
        novos_snapshots[p] = novo_l

    return novos_snapshots


def create_matrix_temporal(data, sps, n):
    y = list()
    values = list()
    cols = list()
    cols.append("Patient_ID")
    for p in sps.keys():
        tp = data[p]
        for snaps in sps[p]:
            l = list()
            l.append(p)

            for e in snaps:
                i = e[0]
                l.extend([tp[i][feature]
                         for feature in tp[i].keys() if feature != "Evolution"])

            values.append(l)
            y.append(e[1])

    cols.extend([f"{ti}{feature}" for ti in range(n)
                 for feature in tp[i].keys() if feature != "Evolution"])

    mats = pd.DataFrame(data=values,
                        columns=cols)

    return mats, y


def create_matrix_static(data, sps):
    y = list()
    values = list()
    cols = list()
    cols.append("Patient_ID")
    for p in sps.keys():
        tp = data[p]
        for snaps in sps[p]:
            l = list()
            l.append(p)
            i = snaps[0][0]
            l.extend([tp[i][feature]
                      for feature in tp[i].keys() if feature != "Evolution"])

            values.append(l)
            y.append(snaps[-1][1])

    cols.extend([f"{feature}"
                 for feature in tp[i].keys() if feature != "Evolution"])

    mats = pd.DataFrame(data=values,
                        columns=cols)

    return mats, y


def create_matrix_temporal_old(data, sps, n, group=False):
    y = list()
    values = list()
    cols = list()
    for p in sps.keys():
        tp = data[p]
        for snaps in sps[p]:
            l = list()
            l.append(p)
            for e in snaps:
                i = e[0]
                l.append(tp[i]['ALS-FRS'])
                l.append(tp[i]['ALS-FRS-R'])
                l.append(tp[i]['ALS-FRSb'])
                l.append(tp[i]['ALS-FRSsUL'])
                l.append(tp[i]['ALS-FRSsLL'])
                l.append(tp[i]['ALS-FRSr'])
                l.append(tp[i]['R'])
                l.append(tp[i]['FVC'])
                l.append(tp[i]['MIP'])
                l.append(tp[i]['MEP'])
                if not group:
                    l.append(tp[i]['prog_rate'])

            values.append(l)
            y.append(e[1])

    cols.append("Patient_ID")
    for ti in range(0, n):
        cols.append(str(ti)+'ALS-FRS')
        cols.append(str(ti)+'ALS-FRS-R')
        cols.append(str(ti)+'ALS-FRSb')
        cols.append(str(ti)+'ALS-FRSsUL')
        cols.append(str(ti)+'ALS-FRSsLL')
        cols.append(str(ti)+'ALS-FRSr')
        cols.append(str(ti)+'R')
        cols.append(str(ti)+'FVC')
        cols.append(str(ti)+'MIP')
        cols.append(str(ti)+'MEP')
        if not group:
            cols.append(str(ti)+'prog_rate')

    mats = pd.DataFrame(data=values,
                        columns=cols)

    return mats, y


def create_matrix_static_old(data, sps, n):
    y = list()
    values = list()
    cols = list()
    for p in sps.keys():
        tp = data[p]
        for snaps in sps[p]:
            l = list()
            l.append(p)
            for e in snaps:
                i = e[0]
                l.append(int(tp[i]['Gender']))
                l.append(float(tp[i]['BMI']))
                l.append(float(tp[i]['MND familiar history']))
                l.append(float(tp[i]['Age at onset']))
                l.append(float(tp[i]['Disease duration']))
                l.append(str(tp[i]['El Escorial reviewed criteria']))
                l.append(str(tp[i]['UMN vs LMN']))
                l.append(str(tp[i]['Onset form']))
                l.append(str(tp[i]['C9orf72']))

            values.append(l)
            y.append(e[1])

    cols.append("Patient_ID")
    for ti in range(0, n):
        cols.append(str(ti)+'Gender')
        cols.append(str(ti)+'BMI')
        cols.append(str(ti)+'MND familiar history')
        cols.append(str(ti)+'Age at onset')
        cols.append(str(ti)+'Disease duration')
        cols.append(str(ti)+'El Escorial reviewed criteria')
        cols.append(str(ti)+'UMN vs LMN')
        cols.append(str(ti)+'Onset form')
        cols.append(str(ti)+'C9orf72')

    mats = pd.DataFrame(data=values,
                        columns=cols)
    return mats, y


def create_mitos_p(data, sps, n):
    alsqs = list()
    for p in sps.keys():
        tp = data[p]
        for snaps in sps[p]:
            lp = list()
            lp.append(p)
            for e in snaps:
                i = e[0]
                lp.append(tp[i]['P1'])
                lp.append(tp[i]['P3'])
                lp.append(tp[i]['P4'])
                lp.append(tp[i]['P6'])
                lp.append(tp[i]['P8'])
                lp.append(tp[i]['1R'])
                lp.append(tp[i]['3R'])

            alsqs.append(lp)

    m_cols = list()
    m_cols.append("Patient_ID")
    for ti in range(0, n):
        m_cols.append("T" + str(ti)+'_P1')
        m_cols.append("T" + str(ti)+'_P3')
        m_cols.append("T" + str(ti)+'_P4')
        m_cols.append("T" + str(ti)+'_P6')
        m_cols.append("T" + str(ti)+'_P8')
        m_cols.append("T" + str(ti)+'_1R')
        m_cols.append("T" + str(ti)+'_3R')

    df_qs = pd.DataFrame(data=alsqs, columns=m_cols)
    return df_qs


def compute_mitos(df_qs, mats, n, last=True):
    r = range(n)
    if last:
        r = range(n-1, n)
    for ti in r:
        m = '{}Movement'.format(ti)
        s = '{}Swallowing'.format(ti)
        c = '{}Communicating'.format(ti)
        b = '{}Breathing'.format(ti)
        ts = 'T{}'.format(ti)

        df_qs[m] = np.where((df_qs[ts + '_P8'].isna()) | (df_qs[ts + '_P6'].isna()),
                            np.nan, np.where((df_qs[ts + '_P8'] > 0) | (df_qs[ts + '_P6'] > 0), 1, 0))
        df_qs[s] = np.where((df_qs[ts + '_P8'].isna()) | (df_qs[ts + '_P6'].isna()),
                            np.nan, np.where((df_qs[ts + '_P3'] > 0), 1, 0))
        df_qs[c] = np.where((df_qs[ts + '_P8'].isna()) | (df_qs[ts + '_P6'].isna()),
                            np.nan, np.where((df_qs[ts + '_P1'] > 0) & (df_qs[ts + '_P4'] > 0), 1, 0))
        df_qs[b] = np.where((df_qs[ts + '_P8'].isna()) | (df_qs[ts + '_P6'].isna()),
                            np.nan, np.where((df_qs[ts + '_1R'] > 0) | (df_qs[ts + '_3R'] > 0), 1, 0))
        mats[str(ti) + 'mitos_s'] = df_qs[m] + df_qs[s] + df_qs[c] + df_qs[b]
    return mats
