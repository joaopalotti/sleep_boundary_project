import os
import pandas as pd
from pycaret.classification import *

def get_env_var(varname, default):
    return int(os.environ.get(varname)) if os.environ.get(varname) is not None else default

def chunks(l, n):
    n = len(l) // n
    return [l[i:i + n] for i in range(0, len(l), max(1, n))]


def data_exists(datafolder, suffix):
    if not os.path.exists(datafolder):
        return False

    for f in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        if not os.path.exists("%s/%s_%s.csv.gz" % (datafolder, f, suffix)):
            return False

    return True

def pycater_setup(train_data, test_data,
                  gt_label="label_5min",
                  ignore_feat=["id", "fold", "linetime", "activity", "percentage_ground_truth"],
                  use_gpu=False):

    numeric_features = list(set(train_data.keys()) - set(ignore_feat) - set([gt_label]))
    experiment = setup(data=train_data, test_data=test_data,
                       target=gt_label, session_id=123,
                       normalize=True, transformation=False,
                       fold_strategy="groupkfold", fold_groups="fold",
                       ignore_features=ignore_feat,
                       silent=False, use_gpu=use_gpu,
                       normalize_method='robust',
                       polynomial_features=False,
                       numeric_features=numeric_features,
                       remove_perfect_collinearity=False,
                       )
    return experiment


# +

def create_xy(train_path="train_data.csv.gz", test_path="test_data.csv.gz", use_gpu=False):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    test_pids = test_data[["pid", "gt_time"]]
    features_to_ignore = ["pid", "fold", "gt_time",
                          "percentage_ground_truth"] if "percentage_ground_truth" in train_data else ["pid", "fold",
                                                                                                      "gt_time"]
    features_to_ignore += ["time_sin", "time_cos"]

    experiment = pycater_setup(train_data, test_data,
                               gt_label="ground_truth",
                               ignore_feat=features_to_ignore,
                               use_gpu=use_gpu)

    # Now we extract the postprocessed features
    # and append back the fold numbers to further break it into train/val
    X_train = get_config("X_train")
    X_test = get_config("X_test")

    y_train = get_config("y_train")
    y_test = get_config("y_test")

    if isinstance(y_test, pd.Series):
        y_test = y_test.to_frame()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame()

    y_train['percentage_ground_truth'] = train_data['percentage_ground_truth']
    y_test['percentage_ground_truth'] = test_data['percentage_ground_truth']

    X_train = pd.concat([train_data["fold"], X_train], axis=1)
    y_train = pd.concat([train_data["fold"], y_train], axis=1)

    # Data is well distributed across the folds
    for f in X_train["fold"].unique():
        print(f, X_train[X_train["fold"] == f].shape)

    X_val = X_train[X_train["fold"] == 9]
    X_train = X_train[X_train["fold"] != 9]

    y_val = y_train[y_train["fold"] == 9]
    y_train = y_train[y_train["fold"] != 9]

    del X_train["fold"], X_val["fold"], y_val["fold"], y_train["fold"]

    X = {"train": X_train, "val": X_val, "test": X_test}
    Y = {"train": y_train, "val": y_val, "test": y_test}

    return X, Y, test_pids


def save_data(output_folder, X, Y, test_pids, featset):
    X["train"].to_csv("%s/X_train_%s.csv.gz" % (output_folder, featset), index=False)
    X["val"].to_csv("%s/X_val_%s.csv.gz" % (output_folder, featset), index=False)
    X["test"].to_csv("%s/X_test_%s.csv.gz" % (output_folder, featset), index=False)

    Y["train"].to_csv("%s/y_train_%s.csv.gz" % (output_folder, featset), index=False)
    Y["val"].to_csv("%s/y_val_%s.csv.gz" % (output_folder, featset), index=False)
    Y["test"].to_csv("%s/y_test_%s.csv.gz" % (output_folder, featset), index=False)

    test_pids.to_csv("%s/test_pids_%s.csv.gz" % (output_folder, featset))


def load_data(datafolder, featset):
    X, Y = {}, {}
    X["train"] = pd.read_csv("%s/X_train_%s.csv.gz" % (datafolder, featset))
    X["val"] = pd.read_csv("%s/X_val_%s.csv.gz" % (datafolder, featset))
    X["test"] = pd.read_csv("%s/X_test_%s.csv.gz" % (datafolder, featset))

    Y["train"] = pd.read_csv("%s/y_train_%s.csv.gz" % (datafolder, featset))
    Y["val"] = pd.read_csv("%s/y_val_%s.csv.gz" % (datafolder, featset))
    Y["test"] = pd.read_csv("%s/y_test_%s.csv.gz" % (datafolder, featset))

    test_pids = pd.read_csv("%s/test_pids_%s.csv.gz" % (datafolder, featset))

    return X, Y, test_pids


if __name__ == "__main__":

    combinations = []
    for win in ["10min_centered", "20min_centered", "40min_centered", "10min_notcentered",
                    "20min_notcentered", "40min_notcentered"]:
        for featset in ["tsfresh", "raw"]:
            combinations.append([win, featset])

    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)

    print("Total combinations:", len(combinations))
    print("All combinations:", combinations)
    selected_combinations = chunks(combinations, SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
    print("Processing: ", selected_combinations)


    for combination in selected_combinations:
        win, featset = combination
        datafolder = "../data/processed/train_test_splits/%s/" % win

        if data_exists(datafolder, featset):
            print("Data already exist at %s. We are done!" % datafolder)
        else:
            print("Creating data....Win: %s, Featset: %s" % (win, featset))
            X, Y, test_pids = create_xy(os.path.join(datafolder, "train_%s_data.csv.gz" % (featset)),
                                        os.path.join(datafolder, "test_%s_data.csv.gz" % featset), use_gpu=True)
            save_data(datafolder, X, Y, test_pids, featset)
