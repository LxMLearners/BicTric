import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, roc_auc_score
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
import constants
from get_patterns import Patterns, Patterns2D
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


def sens(y_true, y_pred): return tp(y_true, y_pred) / \
    (fn(y_true, y_pred) + tp(y_true, y_pred))


def spec(y_true, y_pred): return tn(y_true, y_pred) / \
    (fp(y_true, y_pred) + tn(y_true, y_pred))


def compute_predictions_baseline(n, path, outfile, lastpoint=False):
    directory = os.fsencode(path)
    table = {"File": [], "Model": [], "AUC": [], "AUC_Std": [], "CA": [],
             "CA_Std": [], "Sens": [], "Sens_Std": [], "Spec": [], "Spec_Std": []}
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):

            cts = pd.read_csv(path + '/' + filename)
            y = cts[constants.TARGET].values
            cts.drop(columns=[constants.TARGET], inplace=True)
            if lastpoint:
                feats = set(map(lambda x: x[1:], cts.columns))
                feats_last = list(map(lambda z: str(n-1) + z, feats))
                print(feats_last)
                cts = cts[feats_last]

            X = cts.iloc[:, 0:len(cts.columns)].values

            rskf = RepeatedStratifiedKFold(
                n_splits=10, n_repeats=5, random_state=36851234)
            models = [GaussianNB(), SVC(), xgb.XGBClassifier(eval_metric='logloss', random_state=42,
                                                             use_label_encoder=False), RandomForestClassifier(n_estimators=100, max_depth=7500, random_state=0)]

            scoring = {'accuracy': make_scorer(
                accuracy_score), 'roc_auc': 'roc_auc', 'sens': make_scorer(sens), 'spec': make_scorer(spec)}
            for model in models:
                if isinstance(model, xgb.XGBClassifier):
                    y = list(map(lambda label: 0 if label == 'N' else 1, y))
                scores = cross_validate(model, X, y, scoring=scoring, cv=rskf)

                print("#"*10)
                print()
                print(" =======", str(n) + "TPS", "=======")
                print("File: " + filename)
                table["File"].append(filename)

                print("Classifier: " + type(model).__name__)
                table["Model"].append(type(model).__name__)

                print("AUC: ", scores['test_roc_auc'].mean(),
                      "+-", scores['test_roc_auc'].std())
                table["AUC"].append(scores['test_roc_auc'].mean())
                table["AUC_Std"].append(scores['test_roc_auc'].std())

                print("CA: ", scores['test_accuracy'].mean(),
                      "+-", scores['test_accuracy'].std())
                table["CA"].append(scores['test_accuracy'].mean())
                table["CA_Std"].append(scores['test_accuracy'].std())

                print("Sensitivity: ", scores['test_sens'].mean(
                ), "+-", scores['test_sens'].std())
                table["Sens"].append(scores['test_sens'].mean())
                table["Sens_Std"].append(scores['test_sens'].std())

                print("Specificity: ", scores['test_spec'].mean(
                ), "+-", scores['test_spec'].std())
                table["Spec"].append(scores['test_spec'].mean())
                table["Spec_Std"].append(scores['test_spec'].std())

                if isinstance(model, xgb.XGBClassifier) or isinstance(model, RandomForestClassifier):
                    print("Feature Importance:")
                    model.fit(X, y)
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    # Print the feature ranking (top20)
                    final = 20
                    if len(indices) < 20:
                        final = len(indices)
                    for f in range(final):
                        print("%d. feature %s (%f)" % (
                            f + 1, cts.columns[0:len(cts.columns)][indices[f]], importances[indices[f]]))

    pd.DataFrame(table).to_csv(outfile, index=False)


def compute_predictions(n, path_bics, path_trics, triclusters, biclusters, outfile, feat_selec=False):

    directory_trics = os.fsencode(path_trics)
    directory_bics = os.fsencode(path_bics)

    table = {"File Triclust": [], "File Biclust": [], "Model": [], "AUC": [], "AUC_Std": [
    ], "CA": [], "CA_Std": [], "Sens": [], "Sens_Std": [], "Spec": [], "Spec_Std": []}

    for file_t in sorted(os.listdir(directory_trics)):
        filename_t = os.fsdecode(file_t)
        if filename_t.endswith(".csv"):
            cts_tr = pd.read_csv(path_trics + '/' + filename_t)

            listsf = sorted(os.listdir(directory_bics))

            files_b = listsf
            cts_tr.drop(columns=[constants.TARGET], inplace=True)
            for file_b in files_b:
                filename_b = os.fsdecode(file_b)
                if filename_b.endswith(".csv"):

                    cts_bic = pd.read_csv(path_bics + '/' + filename_b)

                    y = cts_bic[constants.TARGET].values

                    cts_bic.drop(columns=[constants.TARGET], inplace=True)

                    X_t = cts_tr.iloc[:, 1:len(cts_tr.columns)].values
                    X_b = cts_bic.iloc[:, 1:len(cts_bic.columns)].values

                    rskf = RepeatedStratifiedKFold(
                        n_splits=10, n_repeats=5, random_state=36851234)
                    models = [GaussianNB(), SVC(probability=True), xgb.XGBClassifier(eval_metric='logloss', random_state=42,
                                                                                     use_label_encoder=False), RandomForestClassifier(n_estimators=100, max_depth=7500, random_state=0)]

                    # model = RandomForestClassifier(n_estimators=100, max_depth=7500, random_state=0)
                    for model in models:
                        X = np.concatenate((X_t, X_b), axis=1)
                        if isinstance(model, xgb.XGBClassifier):
                            y = np.array(
                                list(map(lambda label: 0 if label == 'N' else 1, y)))
                        if feat_selec and (isinstance(model, xgb.XGBClassifier) or isinstance(model, RandomForestClassifier)):
                            print("N feat before:", len(X[0]))
                            selector = RFECV(model, step=1, cv=10)
                            X = selector.fit_transform(X, y)
                            print("N feat after:", len(X[0]))

                        scores = cross_validate_balancing(model, X, y, folds=5)

                        print("#"*10)
                        print()
                        print(" =======", str(n) + "TPS", "=======")
                        print("Files: " + filename_t + " + " + filename_b)
                        table["File Triclust"].append(filename_t)
                        table["File Biclust"].append(filename_b)

                        print("Classifier: " + type(model).__name__)
                        table["Model"].append(type(model).__name__)

                        print("AUC: ", scores['test_roc_auc'].mean(
                        ), "+-", scores['test_roc_auc'].std())
                        table["AUC"].append(scores['test_roc_auc'].mean())
                        table["AUC_Std"].append(scores['test_roc_auc'].std())

                        print("CA: ", scores['test_accuracy'].mean(
                        ), "+-", scores['test_accuracy'].std())
                        table["CA"].append(scores['test_accuracy'].mean())
                        table["CA_Std"].append(scores['test_accuracy'].std())

                        print("Sensitivity: ", scores['test_sens'].mean(
                        ), "+-", scores['test_sens'].std())
                        table["Sens"].append(scores['test_sens'].mean())
                        table["Sens_Std"].append(scores['test_sens'].std())

                        print("Specificity: ", scores['test_spec'].mean(
                        ), "+-", scores['test_spec'].std())
                        table["Spec"].append(scores['test_spec'].mean())
                        table["Spec_Std"].append(scores['test_spec'].std())

                        if isinstance(model, xgb.XGBClassifier) or isinstance(model, RandomForestClassifier):
                            print("Feature Importance:")
                            model.fit(X, y)
                            importances = model.feature_importances_
                            indices = np.argsort(importances)[::-1]

                            cols_bics = list(cts_bic.columns)
                            #cols_bics = list(map(lambda c: c.replace("Tric","Bic"), cts_bic.columns))
                            columns = list(cts_tr.columns[1:len(
                                cts_tr.columns)]) + cols_bics[1:len(cols_bics)]
                            # Print the feature ranking (top20)
                            final = 20
                            if len(indices) < 20:
                                final = len(indices)

                            p2 = Patterns2D()
                            p2.load_biclusters(biclusters)

                            p = Patterns()
                            p.load_triclusters(triclusters)

                            for f in range(final):
                                t_name = columns[indices[f]]
                                t_id = int(t_name.split("_")[1])
                                t_ix = int(t_name.split("_")[2])
                                if t_name.split("_")[0] == 'Bic':
                                    pattern = p2.print_pattern_i(t_id, t_ix)
                                else:
                                    pattern = p.print_pattern_i(t_id, t_ix)

                                print("%d. feature %s (%f)" % (
                                    f + 1, pattern, importances[indices[f]]))

    pd.DataFrame(table).to_csv(outfile, index=False)


def compute_predictions_triclustering(n, path_trics, triclusters, outfile):
    directory = os.fsencode(path_trics)
    table = {"File": [], "Model": [], "AUC": [], "AUC_Std": [], "CA": [],
             "CA_Std": [], "Sens": [], "Sens_Std": [], "Spec": [], "Spec_Std": []}
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):

            cts = pd.read_csv(path_trics + '/' + filename)
            y = cts[constants.TARGET].values

            cts.drop(columns=[constants.TARGET], inplace=True)
            X = cts.iloc[:, :len(cts.columns)].values

            print(pd.Series(y).value_counts())
            models = [GaussianNB(), SVC(probability=True), xgb.XGBClassifier(eval_metric='logloss', random_state=42,
                                                                             use_label_encoder=False), RandomForestClassifier(n_estimators=100, max_depth=7500, random_state=0)]
            for model in models:
                # model = RandomForestClassifier(n_estimators=100, max_depth=7500, random_state=0)
                if isinstance(model, xgb.XGBClassifier):
                    y = np.array(
                        list(map(lambda label: 0 if label == 'N' else 1, y)))
                # scoring = {'accuracy': make_scorer(accuracy_score), 'roc_auc': 'roc_auc', 'sens': make_scorer(sens), 'spec': make_scorer(spec)}
                scores = cross_validate_balancing(model, X, y, folds=10)

                print("#"*10)
                print()
                print(" =======", str(n) + "TPS", "=======")
                print("File: " + filename)
                table["File"].append(filename)

                print("Classifier: " + type(model).__name__)
                table["Model"].append(type(model).__name__)

                print("AUC: ", scores['test_roc_auc'].mean(),
                      "+-", scores['test_roc_auc'].std())
                table["AUC"].append(scores['test_roc_auc'].mean())
                table["AUC_Std"].append(scores['test_roc_auc'].std())

                print("CA: ", scores['test_accuracy'].mean(),
                      "+-", scores['test_accuracy'].std())
                table["CA"].append(scores['test_accuracy'].mean())
                table["CA_Std"].append(scores['test_accuracy'].std())

                print("Sensitivity: ", scores['test_sens'].mean(
                ), "+-", scores['test_sens'].std())
                table["Sens"].append(scores['test_sens'].mean())
                table["Sens_Std"].append(scores['test_sens'].std())

                print("Specificity: ", scores['test_spec'].mean(
                ), "+-", scores['test_spec'].std())
                table["Spec"].append(scores['test_spec'].mean())
                table["Spec_Std"].append(scores['test_spec'].std())

                if isinstance(model, xgb.XGBClassifier) or isinstance(model, RandomForestClassifier):
                    print("Feature Importance:")
                    model.fit(X, y)
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    # Print the feature ranking (top20)

                    final = 20
                    if len(indices) < 20:
                        final = len(indices)

                    p = Patterns()
                    p.load_triclusters(triclusters)
                    columns = cts.columns
                    for f in range(final):
                        t_name = columns[indices[f]]
                        t_id = int(t_name.split("_")[1])
                        t_ix = int(t_name.split("_")[2])

                        pattern = p.print_pattern_i(t_id, t_ix)

                        print("%d. feature %s (%f)" % (
                            f + 1, pattern, importances[indices[f]]))
    pd.DataFrame(table).to_csv(outfile, index=False)


def cross_validate_balancing_and_groups(estimator, X, y, groups, balance=SMOTE(random_state=0,  k_neighbors=3), folds=10):
    """Function that takes a sklearn classifier (estimator), 
    X and y, splits using StratifiedKFols with the number of folds defined, 
    balances just the train data for fit and uses unbalanced but stratified test set to predict.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    balance: balancing instance from imblearn, default=SMOTE(). 
        If 'None', no balancing is done 

    folds : int, default=10
        Number of folds. Must be at least 2.

    Returns
    -------
    scores : dict of float arrays of shape (folds,)
        Array of scores of the estimator for each run of the cross validation.
    """

    cv = RepeatedStratifiedKFold(n_splits=folds, random_state=0, n_repeats=5)

    scores_groups = {
        'slow': {'AUC': [], 'CA': [], 'Sens': [], 'Spec': []},
        'normal': {'AUC': [], 'CA': [], 'Sens': [], 'Spec': []},
        'fast': {'AUC': [], 'CA': [], 'Sens': [], 'Spec': []}
    }
    scores = {
        'test_accuracy': [],
        'test_roc_auc': [],
        'test_sens': [],
        'test_spec': []}

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if balance != 'None':
            rus = RandomUnderSampler(random_state=0, sampling_strategy=0.5)
            X_train, y_train = rus.fit_resample(X_train, y_train)
            X_train, y_train = balance.fit_resample(X_train, y_train)

        estimator.fit(X_train, y_train)
        y_predicted = estimator.predict(X_test)

        scores['test_accuracy'].append(accuracy_score(y_test, y_predicted))
        scores['test_roc_auc'].append(roc_auc_score(
            y_test, estimator.predict_proba(X_test)[:, 1]))
        scores['test_sens'].append(sens(y_test, y_predicted))
        scores['test_spec'].append(spec(y_test, y_predicted))

        ##### EVALUATION FOR GROUPS ######
        groups_test = np.array(groups)[test_idx.astype(int)]
        y_slow = list()
        y_norm = list()
        y_fast = list()

        y_pred_slow = list()
        y_pred_norm = list()
        y_pred_fast = list()

        for i in range(len(groups_test)):
            if groups_test[i] == 'slow':
                y_slow.append(y_test[i])
                y_pred_slow.append(y_predicted[i])
            elif groups_test[i] == 'normal':
                y_norm.append(y_test[i])
                y_pred_norm.append(y_predicted[i])
            else:
                y_fast.append(y_test[i])
                y_pred_fast.append(y_predicted[i])

        y_slow = np.array(y_slow)
        y_norm = np.array(y_norm)
        y_fast = np.array(y_fast)

        y_pred_slow = np.array(y_pred_slow)
        y_pred_norm = np.array(y_pred_norm)
        y_pred_fast = np.array(y_pred_fast)

        # scores_groups["slow"]['AUC'].append(roc_auc_score(y_slow, estimator.predict_proba(X_test)[:, 1]))
        scores_groups["slow"]['CA'].append(accuracy_score(y_slow, y_pred_slow))
        scores_groups["slow"]['Sens'].append(sens(y_slow, y_pred_slow))
        scores_groups["slow"]['Spec'].append(spec(y_slow, y_pred_slow))

        # scores_groups["normal"]['AUC'].append(roc_auc_score(y_norm, estimator.predict_proba(X_test)[:, 1]))
        scores_groups["normal"]['CA'].append(
            accuracy_score(y_norm, y_pred_norm))
        scores_groups["normal"]['Sens'].append(sens(y_norm, y_pred_norm))
        scores_groups["normal"]['Spec'].append(spec(y_norm, y_pred_norm))

        # scores_groups["fast"]['AUC'].append(roc_auc_score(y_norm, estimator.predict_proba(X_test)[:, 1]))
        scores_groups["fast"]['CA'].append(accuracy_score(y_fast, y_pred_fast))
        scores_groups["fast"]['Sens'].append(sens(y_fast, y_pred_fast))
        scores_groups["fast"]['Spec'].append(spec(y_fast, y_pred_fast))

    scores['test_accuracy'] = np.array(scores['test_accuracy'])
    scores['test_roc_auc'] = np.array(scores['test_roc_auc'])
    scores['test_sens'] = np.array(scores['test_sens'])
    scores['test_spec'] = np.array(scores['test_spec'])

    for g in ['slow', 'normal', 'fast']:
        scores_groups[g]['CA'] = np.array(scores_groups[g]['CA'])
        scores_groups[g]['Sens'] = np.array(scores_groups[g]['Sens'])
        scores_groups[g]['Spec'] = np.array(scores_groups[g]['Spec'])

    return scores, scores_groups


def cross_validate_balancing(estimator, X, y, balance=SMOTE(random_state=0,  k_neighbors=3), folds=10):
    """Function that takes a sklearn classifier (estimator), 
    X and y, splits using StratifiedKFols with the number of folds defined, 
    balances just the train data for fit and uses unbalanced but stratified test set to predict.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    balance: balancing instance from imblearn, default=SMOTE(). 
        If 'None', no balancing is done 

    folds : int, default=10
        Number of folds. Must be at least 2.

    Returns
    -------
    scores : dict of float arrays of shape (folds,)
        Array of scores of the estimator for each run of the cross validation.
    """

    cv = RepeatedStratifiedKFold(n_splits=folds, random_state=0, n_repeats=5)

    scores = {
        'test_accuracy': [],
        'test_roc_auc': [],
        'test_sens': [],
        'test_spec': []}

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if balance != 'None':
            rus = RandomUnderSampler(random_state=0, sampling_strategy=0.5)
            try:
                X_train, y_train = rus.fit_resample(X_train, y_train)
            except:
                pass
            X_train, y_train = balance.fit_resample(X_train, y_train)

        estimator.fit(X_train, y_train)
        y_predicted = estimator.predict(X_test)

        scores['test_accuracy'].append(accuracy_score(y_test, y_predicted))
        scores['test_roc_auc'].append(roc_auc_score(
            y_test, estimator.predict_proba(X_test)[:, 1]))
        scores['test_sens'].append(sens(y_test, y_predicted))
        scores['test_spec'].append(spec(y_test, y_predicted))

    scores['test_accuracy'] = np.array(scores['test_accuracy'])
    scores['test_roc_auc'] = np.array(scores['test_roc_auc'])
    scores['test_sens'] = np.array(scores['test_sens'])
    scores['test_spec'] = np.array(scores['test_spec'])

    return scores
