# DATACORTECH TabICL evaluation
# Mirrors the existing DATACORTECH_model_features_eda.ipynb protocol and swaps TabPFN -> TabICL.

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, cohen_kappa_score, matthews_corrcoef, roc_auc_score
from tabicl import TabICLClassifier

start_time = datetime.now()

data_path = "/home/fcolanto/projects/pfn-augmentation/data/Datacortech_final_data.xlsx"
parser = argparse.ArgumentParser()
parser.add_argument("--local-ckpt-path", type=str, default=None)
parser.add_argument("--local-model-label", type=str, default="TabICL_local")
args = parser.parse_args()

local_ckpt_path = args.local_ckpt_path


datacor = pd.read_excel(data_path)
datacor.columns = datacor.columns.astype(str).str.strip()
datacor["Efficiency"] = pd.Categorical(datacor["Efficiency"], categories=["Inhibitor", "No"])
datacor["pH_2"] = pd.Categorical(datacor["pH_2"], categories=["acidic", "neutral", "basic"])
datacor = pd.DataFrame(datacor)

datacor_all = datacor.copy()
np.random.seed(12345)

ph_dummies = pd.get_dummies(datacor_all["pH_2"], prefix="pH_2").astype(int)
datacor_all = pd.concat([datacor_all, ph_dummies], axis=1)
datacor_all = pd.DataFrame(datacor_all)

cv_idx = np.r_[
    0:80, 100:180, 200:280, 300:380, 400:480, 500:580, 600:680, 700:780,
    800:880, 900:980, 1000:1080, 1100:1180, 1200:1280, 1300:1380,
    1400:1480, 1500:1580, 1600:1680, 1700:1880, 1900:1966
]
test_idx = np.r_[
    80:100, 180:200, 280:300, 380:400, 480:500, 580:600, 680:700, 780:800,
    880:900, 980:1000, 1080:1100, 1180:1200, 1280:1300, 1380:1400,
    1480:1500, 1580:1600, 1680:1700, 1880:1900
]

datacor_all_CV = datacor_all.iloc[cv_idx].copy()
datacor_all_test = datacor_all.iloc[test_idx].copy()

nrFolds = 10
folds = np.resize(np.repeat(np.arange(1, nrFolds + 1), 10), len(datacor_all_CV))

feature_cols = [
    "pH_2_neutral",
    "ALogP",
    "tpsaEfficiency",
    "bpol",
    "apol",
    "WTPT.5",
    "ALogp2",
    "ATSm1",
    "WTPT.3",
    "XLogP",
]
target_col = "Efficiency"
tabicl_cutoff = 0.45

def classification_metrics(y_true, y_pred, y_prob=None, positive_label="Inhibitor"):
    labels = [positive_label, "No"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    accuracy = (tp + tn) / cm.sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    balanced_accuracy = (sensitivity + specificity) / 2
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    matthews = matthews_corrcoef(
        pd.Series(y_true).map({positive_label: 1, "No": 0}),
        pd.Series(y_pred).map({positive_label: 1, "No": 0}),
    )
    y_true_bin = pd.Series(y_true).map({positive_label: 1, "No": 0}).to_numpy()
    auroc = roc_auc_score(y_true_bin, y_prob) if y_prob is not None else np.nan
    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Balanced Accuracy": balanced_accuracy,
        "Kappa": kappa,
        "Matthews": matthews,
        "AUROC": auroc,
    }

def get_inhibitor_probability(model, X, positive_label="Inhibitor"):
    prob = model.predict_proba(X)
    class_list = list(model.classes_)
    positive_idx = class_list.index(positive_label)
    return prob[:, positive_idx]

def run_tabicl_eval(model_label, model_path=None, checkpoint_version="tabicl-classifier-v2-20260212.ckpt", device="cuda", allow_auto_download=True):
    accuracy_cv = np.empty(nrFolds, dtype=float)
    sensitivity_cv = np.empty(nrFolds, dtype=float)
    specificity_cv = np.empty(nrFolds, dtype=float)
    balanced_accuracy_cv = np.empty(nrFolds, dtype=float)
    kappa_cv = np.empty(nrFolds, dtype=float)
    matthews_cv = np.empty(nrFolds, dtype=float)
    auroc_cv = np.empty(nrFolds, dtype=float)

    for k in range(1, nrFolds + 1):
        fold = np.where(folds == k)[0]
        datacor_train_CV = datacor_all_CV.drop(datacor_all_CV.index[fold]).copy()
        datacor_test_CV = datacor_all_CV.iloc[fold].copy()

        X_train = datacor_train_CV[feature_cols]
        y_train = datacor_train_CV[target_col].astype(str)
        X_valid = datacor_test_CV[feature_cols]
        y_valid = datacor_test_CV[target_col].astype(str)

        tabicl_cv = TabICLClassifier(
            device=device,
            model_path=model_path,
            checkpoint_version=checkpoint_version,
            allow_auto_download=allow_auto_download,
        )
        tabicl_cv.fit(X_train, y_train)

        inhibitor_prob = get_inhibitor_probability(tabicl_cv, X_valid, positive_label="Inhibitor")
        y_pred = np.where(inhibitor_prob >= tabicl_cutoff, "Inhibitor", "No")

        result_cv = classification_metrics(y_valid.to_numpy(), y_pred, y_prob=inhibitor_prob, positive_label="Inhibitor")
        accuracy_cv[k - 1] = result_cv["Accuracy"]
        sensitivity_cv[k - 1] = result_cv["Sensitivity"]
        specificity_cv[k - 1] = result_cv["Specificity"]
        balanced_accuracy_cv[k - 1] = result_cv["Balanced Accuracy"]
        kappa_cv[k - 1] = result_cv["Kappa"]
        matthews_cv[k - 1] = result_cv["Matthews"]
        auroc_cv[k - 1] = result_cv["AUROC"]

    X_train_full = datacor_all_CV[feature_cols]
    y_train_full = datacor_all_CV[target_col].astype(str)
    X_test = datacor_all_test[feature_cols]
    y_test = datacor_all_test[target_col].astype(str)

    tabicl_test = TabICLClassifier(
        device=device,
        model_path=model_path,
        checkpoint_version=checkpoint_version,
        allow_auto_download=allow_auto_download,
    )
    tabicl_test.fit(X_train_full, y_train_full)

    inhibitor_prob_test = get_inhibitor_probability(tabicl_test, X_test, positive_label="Inhibitor")
    y_pred_test = np.where(inhibitor_prob_test >= tabicl_cutoff, "Inhibitor", "No")
    result_test = classification_metrics(y_test.to_numpy(), y_pred_test, y_prob=inhibitor_prob_test, positive_label="Inhibitor")

    return {
        "Model": model_label,
        "CV_Accuracy_mean": np.mean(accuracy_cv),
        "CV_Accuracy_2sd": 2 * np.std(accuracy_cv, ddof=1),
        "CV_Sensitivity_mean": np.mean(sensitivity_cv),
        "CV_Sensitivity_2sd": 2 * np.std(sensitivity_cv, ddof=1),
        "CV_Specificity_mean": np.mean(specificity_cv),
        "CV_Specificity_2sd": 2 * np.std(specificity_cv, ddof=1),
        "CV_BalAcc_mean": np.mean(balanced_accuracy_cv),
        "CV_BalAcc_2sd": 2 * np.std(balanced_accuracy_cv, ddof=1),
        "CV_Kappa_mean": np.mean(kappa_cv),
        "CV_Kappa_2sd": 2 * np.std(kappa_cv, ddof=1),
        "CV_Matthews_mean": np.mean(matthews_cv),
        "CV_Matthews_2sd": 2 * np.std(matthews_cv, ddof=1),
        "CV_AUROC_mean": np.mean(auroc_cv),
        "CV_AUROC_2sd": 2 * np.std(auroc_cv, ddof=1),
        "Test_Accuracy": result_test["Accuracy"],
        "Test_Sensitivity": result_test["Sensitivity"],
        "Test_Specificity": result_test["Specificity"],
        "Test_BalAcc": result_test["Balanced Accuracy"],
        "Test_Kappa": result_test["Kappa"],
        "Test_Matthews": result_test["Matthews"],
        "Test_AUROC": result_test["AUROC"],
    }

pretrained_tabicl_summary = run_tabicl_eval(
    model_label="TabICL_pretrained_v2",
    model_path=None,
    checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
    device="cuda",
    allow_auto_download=True,
)

local_tabicl_summary = None
if local_ckpt_path is not None:
    local_tabicl_summary = run_tabicl_eval(
        model_label=args.local_model_label,
        model_path=local_ckpt_path,
        checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
        device="cuda",
        allow_auto_download=False,
    )

results = [pretrained_tabicl_summary]
if local_tabicl_summary is not None:
    results.append(local_tabicl_summary)

comparison_df = pd.DataFrame(results).round(3)
print(comparison_df.to_string(index=False))

end_time = datetime.now()
print("Started:", start_time)
print("Finished:", end_time)
print("Duration:", end_time - start_time)
