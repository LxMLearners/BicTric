from os import confstr_names
import sys
import subprocess
from pathlib import Path
from make_predictions import compute_predictions, compute_predictions_triclustering
from tabfiles_utils import tab_from_baseline_temporal, tab_from_baseline_static
from compute_similarities import sim_matrix_tric, sim_matrix_bic
import constants

# Usage: python3 bictric.py <n> <config_file> [<cr_point>] [<tw>] [<group>]
if __name__ == "__main__":

    n = int(sys.argv[1])
    config_file = sys.argv[2]
    cr_point = None
    tw = None
    if len(sys.argv) > 3:
        cr_point = sys.argv[3]
        tw = sys.argv[4]
    constants.get_config(config_file, cr_point, tw)
    group = ""
    if len(sys.argv) > 5:
        group = sys.argv[5]

    top_folder = constants.TOP_FOLDER.format(tw, cr_point)

    baseline_folder_static = constants.BASELINE_DIR_S + "{}TPS/".format(n)
    baseline_folder_temporal = constants.BASELINE_DIR_T + "{}TPS/".format(n)

    tabfiles_folder = constants.TAB_DIR + "{}TPS/".format(n)
    biclusters_folder = constants.BICLUSTERS_DIR + \
        "{}TPS{}".format(n, group)
    triclusters_folder = constants.TRICLUSTERS_DIR + \
        "{}TPS{}".format(n, group)
    matrices_folder_s = constants.MATRICES_DIR_S + \
        "{}TPS{}/".format(n, group)
    matrices_folder_t = constants.MATRICES_DIR_T + \
        "{}TPS{}/".format(n, group)
    results_folder = constants.RESULTS_DIR
    results_folder_tric = constants.RESULTS_DIR_TRIC

    Path(tabfiles_folder).mkdir(parents=True, exist_ok=True)
    Path(biclusters_folder).mkdir(parents=True, exist_ok=True)
    Path(triclusters_folder).mkdir(parents=True, exist_ok=True)
    Path(matrices_folder_s).mkdir(parents=True, exist_ok=True)
    Path(matrices_folder_t).mkdir(parents=True, exist_ok=True)
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    Path(results_folder_tric).mkdir(parents=True, exist_ok=True)

    print("*** Starting BicTric Process ***")
    print("Parameters: ")

    print("Number of Consecutive Snapshots:", n)

    print()
    print("Creating TAB FILES")

    out_bic = "{}TAB_BIC_{}TPS{}.tab".format(tabfiles_folder, n, group)
    out_tri = "{}TAB_{}TPS{}.tab".format(tabfiles_folder, n, group)

    baseline_file_static = baseline_folder_static + \
        "{}TPS{}_baseline_static.csv".format(n, group)
    baseline_file_temporal = baseline_folder_temporal + \
        "{}TPS{}_baseline_temporal.csv".format(n, group)

    tab_from_baseline_static(baseline_file_static,
                                out_bic, list(constants.STATIC_FEATURES.keys()))
    nP = tab_from_baseline_temporal(
        baseline_file_temporal, out_tri, n, list(constants.TEMPORAL_FEATURES.keys()))
    nP = int(nP*0.1)

    print("TAB FILES created!")

    # 1. Run TCtriCluster (biclusters and triclusters)
    # 1.1 Triclustering first
    print("Running Triclustering")
    cmd = "python3 src/TCtriCluster.py -f {}TAB_{}TPS{}.tab -sT 2 -sS 1 -sG {} -w 0.01 -o 1 -mv 0.5 > {}/out_1.txt".format(
        tabfiles_folder, n, group, nP, triclusters_folder)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print("Triclustering Completed")

    # 1.2 Biclustering next
    print("Running Biclustering")

    cmd = "python3 src/TCtriCluster.py -f {}TAB_BIC_{}TPS{}.tab -sT 1 -sS 2 -sG {} -w 0.01 -o 1 -mv 0.5 > {}/out_1.txt".format(
        tabfiles_folder, n, group, nP, biclusters_folder)
    print(cmd)
    subprocess.call(cmd, shell=True)

    print("Biclustering Completed")

    sim_matrix_bic(n, baseline_file_static,
                    biclusters_folder, matrices_folder_s)

    path_trics = top_folder + "triclusters/{}TPS".format(n)
    path_matr_t = top_folder + "matrices/trics/{}TPS/".format(n)

    sim_matrix_tric(n, baseline_file_temporal, triclusters_folder, matrices_folder_t,
                    last=False, tri=constants.PATTERNS_3D)  # tri - Patterns 3D
    
    ## PREDICTIONS
    out_file = "results_{}TPS{}.csv".format(n, group)
    print("### BICTRIC RESULTS ###")

    compute_predictions(n, matrices_folder_s, matrices_folder_t, triclusters_folder + '/out_1.txt', biclusters_folder + '/out_1.txt',
                        results_folder + out_file, feat_selec=constants.FEATURE_SELECTION)

