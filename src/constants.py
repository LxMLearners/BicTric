import yaml as yy
from yaml.loader import Loader


def get_config(config_file, cr_point=None, tw=None, group=None):
    s = open(config_file, 'r')
    cfs = yy.load(s, Loader=Loader)
    globals().update(cfs)

    global BASELINE_DIR_S
    global BASELINE_DIR_T
    global TAB_DIR
    global TRICLUSTERS_DIR
    global BICLUSTERS_DIR
    global MATRICES_DIR_T
    global MATRICES_DIR_S
    global RESULTS_DIR
    global RESULTS_BASELINE_DIR
    global RESULTS_DIR_TRIC
    global TOP_FOLDER
    global DATA_FILE

    if cr_point and tw:
        group = "" if group is None else group
        TOP_FOLDER = TOP_FOLDER.format(tw, cr_point)
        DATA_FILE = DATA_FILE.format(cr_point, tw, group)

    BASELINE_DIR_S = TOP_FOLDER + "baselines/static/"
    BASELINE_DIR_T = TOP_FOLDER + "baselines/temporal/"

    TAB_DIR = TOP_FOLDER + "tab_files/"
    TRICLUSTERS_DIR = TOP_FOLDER + "triclusters/"
    BICLUSTERS_DIR = TOP_FOLDER + "biclusters/"

    MATRICES_DIR_T = TOP_FOLDER + "matrices/trics/"
    MATRICES_DIR_S = TOP_FOLDER + "matrices/bics/"

    RESULTS_DIR = TOP_FOLDER + "results/bictric/"
    RESULTS_DIR_TRIC = TOP_FOLDER + "results/tric-based/"

    if PATTERNS_3D:
        MATRICES_DIR_T = TOP_FOLDER + "matrices/trics3D/"
        RESULTS_DIR = TOP_FOLDER + "results3D/bictric/"

    if FEATURE_SELECTION:
        RESULTS_DIR = TOP_FOLDER + "results/bictric_FS/"

    RESULTS_BASELINE_DIR = TOP_FOLDER + "results/baseline/"
