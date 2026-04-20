# DATACORTECH TabICL evaluation
# Mirrors the existing DATACORTECH_model_features_eda.ipynb protocol and swaps TabPFN -> TabICL.

import argparse
from pathlib import Path
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
    if y_prob is None:
        auroc = np.nan
    else:
        y_prob = np.asarray(y_prob, dtype=float)
        if np.isnan(y_prob).any() or np.isinf(y_prob).any():
            auroc = np.nan
        else:
            auroc = roc_auc_score(y_true_bin, y_prob)

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
    prob = np.asarray(model.predict_proba(X), dtype=float)
    if prob.ndim != 2:
        raise ValueError(f"predict_proba returned array with shape {prob.shape}, expected 2D array")

    class_list = [str(c) for c in model.classes_]
    if positive_label not in class_list:
        raise ValueError(f"Positive label '{positive_label}' not found in model.classes_: {class_list}")

    if prob.shape[1] != len(class_list):
        raise ValueError(
            f"predict_proba returned {prob.shape[1]} columns but model.classes_ has {len(class_list)} entries"
        )

    positive_idx = class_list.index(positive_label)
    return prob[:, positive_idx]


def make_tabicl_classifier(model_path=None, checkpoint_version="tabicl-classifier-v2-20260212.ckpt", device="cuda", allow_auto_download=True):
    if model_path is None:
        return TabICLClassifier(
            device=device,
            checkpoint_version=checkpoint_version,
            allow_auto_download=allow_auto_download,
        )

    return TabICLClassifier(
        device=device,
        model_path=str(model_path),
        allow_auto_download=False,
    )


def validate_probability_vector(prob, model_label, split_name):
    prob = np.asarray(prob, dtype=float)

    if prob.ndim != 1:
        raise ValueError(f"{model_label} {split_name}: expected 1D probability vector, got shape {prob.shape}")

    finite_mask = np.isfinite(prob)
    if not finite_mask.all():
        n_total = int(prob.size)
        n_nan = int(np.isnan(prob).sum())
        n_posinf = int(np.isposinf(prob).sum())
        n_neginf = int(np.isneginf(prob).sum())
        finite_vals = prob[finite_mask]

        if finite_vals.size > 0:
            finite_min = float(finite_vals.min())
            finite_max = float(finite_vals.max())
            finite_info = f"finite_min={finite_min:.6g}, finite_max={finite_max:.6g}"
        else:
            finite_info = "finite_min=NA, finite_max=NA"

        raise ValueError(
            f"{model_label} {split_name}: non-finite probabilities detected "
            f"(total={n_total}, nan={n_nan}, posinf={n_posinf}, neginf={n_neginf}, {finite_info})"
        )

    return prob


def probability_summary(prob, cutoff):
    prob = np.asarray(prob, dtype=float)
    return {
        "min": float(prob.min()),
        "max": float(prob.max()),
        "mean": float(prob.mean()),
        "positive_rate": float((prob >= cutoff).mean()),
    }


def select_best_cutoff(y_true, y_prob, positive_label="Inhibitor", default_cutoff=0.45):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    candidate_cutoffs = np.unique(np.concatenate(([0.0, 1.0, default_cutoff], y_prob)))

    best_cutoff = float(default_cutoff)
    best_bal_acc = -np.inf

    for cutoff in candidate_cutoffs:
        y_pred = np.where(y_prob >= cutoff, positive_label, "No")
        metrics = classification_metrics(y_true, y_pred, y_prob=y_prob, positive_label=positive_label)
        bal_acc = metrics["Balanced Accuracy"]

        if bal_acc > best_bal_acc or (
            np.isclose(bal_acc, best_bal_acc) and abs(cutoff - default_cutoff) < abs(best_cutoff - default_cutoff)
        ):
            best_bal_acc = bal_acc
            best_cutoff = float(cutoff)

    return best_cutoff


def run_tabicl_eval(model_label, model_path=None, checkpoint_version="tabicl-classifier-v2-20260212.ckpt", device="cuda", allow_auto_download=True):
    accuracy_cv = np.empty(nrFolds, dtype=float)
    sensitivity_cv = np.empty(nrFolds, dtype=float)
    specificity_cv = np.empty(nrFolds, dtype=float)
    balanced_accuracy_cv = np.empty(nrFolds, dtype=float)
    kappa_cv = np.empty(nrFolds, dtype=float)
    matthews_cv = np.empty(nrFolds, dtype=float)
    auroc_cv = np.empty(nrFolds, dtype=float)
    cv_prob_list = []
    cv_true_list = []

    for k in range(1, nrFolds + 1):
        fold = np.where(folds == k)[0]
        datacor_train_CV = datacor_all_CV.drop(datacor_all_CV.index[fold]).copy()
        datacor_test_CV = datacor_all_CV.iloc[fold].copy()

        X_train = datacor_train_CV[feature_cols]
        y_train = datacor_train_CV[target_col].astype(str)
        X_valid = datacor_test_CV[feature_cols]
        y_valid = datacor_test_CV[target_col].astype(str)

        tabicl_cv = make_tabicl_classifier(
            model_path=model_path,
            checkpoint_version=checkpoint_version,
            device=device,
            allow_auto_download=allow_auto_download,
        )
        tabicl_cv.fit(X_train, y_train)

        inhibitor_prob = get_inhibitor_probability(tabicl_cv, X_valid, positive_label="Inhibitor")
        inhibitor_prob = validate_probability_vector(inhibitor_prob, model_label, f"CV fold {k}")
        cv_prob_list.append(inhibitor_prob)
        cv_true_list.append(y_valid.to_numpy())
        y_pred = np.where(inhibitor_prob >= tabicl_cutoff, "Inhibitor", "No")

        result_cv = classification_metrics(
            y_valid.to_numpy(),
            y_pred,
            y_prob=inhibitor_prob,
            positive_label="Inhibitor",
        )
        accuracy_cv[k - 1] = result_cv["Accuracy"]
        sensitivity_cv[k - 1] = result_cv["Sensitivity"]
        specificity_cv[k - 1] = result_cv["Specificity"]
        balanced_accuracy_cv[k - 1] = result_cv["Balanced Accuracy"]
        kappa_cv[k - 1] = result_cv["Kappa"]
        matthews_cv[k - 1] = result_cv["Matthews"]
        auroc_cv[k - 1] = result_cv["AUROC"]

    cv_prob_all = np.concatenate(cv_prob_list)
    cv_true_all = np.concatenate(cv_true_list)
    best_cv_cutoff = select_best_cutoff(
        cv_true_all,
        cv_prob_all,
        positive_label="Inhibitor",
        default_cutoff=tabicl_cutoff,
    )
    y_pred_cv_best = np.where(cv_prob_all >= best_cv_cutoff, "Inhibitor", "No")
    cv_best_metrics = classification_metrics(
        cv_true_all,
        y_pred_cv_best,
        y_prob=cv_prob_all,
        positive_label="Inhibitor",
    )

    X_train_full = datacor_all_CV[feature_cols]
    y_train_full = datacor_all_CV[target_col].astype(str)
    X_test = datacor_all_test[feature_cols]
    y_test = datacor_all_test[target_col].astype(str)

    tabicl_test = make_tabicl_classifier(
        model_path=model_path,
        checkpoint_version=checkpoint_version,
        device=device,
        allow_auto_download=allow_auto_download,
    )
    tabicl_test.fit(X_train_full, y_train_full)

    inhibitor_prob_test = get_inhibitor_probability(tabicl_test, X_test, positive_label="Inhibitor")
    inhibitor_prob_test = validate_probability_vector(inhibitor_prob_test, model_label, "test")
    y_pred_test = np.where(inhibitor_prob_test >= tabicl_cutoff, "Inhibitor", "No")
    prob_stats_test = probability_summary(inhibitor_prob_test, tabicl_cutoff)
    prob_stats_test_best = probability_summary(inhibitor_prob_test, best_cv_cutoff)

    result_test = classification_metrics(
        y_test.to_numpy(),
        y_pred_test,
        y_prob=inhibitor_prob_test,
        positive_label="Inhibitor",
    )
    y_pred_test_best = np.where(inhibitor_prob_test >= best_cv_cutoff, "Inhibitor", "No")
    result_test_best = classification_metrics(
        y_test.to_numpy(),
        y_pred_test_best,
        y_prob=inhibitor_prob_test,
        positive_label="Inhibitor",
    )

    model_source = getattr(tabicl_test, "model_path_", None)
    if model_source is None:
        model_source = checkpoint_version
    print(
        f"[{model_label}] source={model_source} cutoff={tabicl_cutoff:.2f} "
        f"test_prob_min={prob_stats_test['min']:.6f} "
        f"test_prob_max={prob_stats_test['max']:.6f} "
        f"test_prob_mean={prob_stats_test['mean']:.6f} "
        f"test_positive_rate={prob_stats_test['positive_rate']:.6f}"
    )
    print(
        f"[{model_label}] cv_best_cutoff={best_cv_cutoff:.6f} "
        f"cv_balacc_at_best_cutoff={cv_best_metrics['Balanced Accuracy']:.6f} "
        f"test_positive_rate_at_best_cutoff={prob_stats_test_best['positive_rate']:.6f}"
    )

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
        "CV_BestCutoff": best_cv_cutoff,
        "CV_BalAcc_bestCutoff": cv_best_metrics["Balanced Accuracy"],
        "Test_Accuracy": result_test["Accuracy"],
        "Test_Sensitivity": result_test["Sensitivity"],
        "Test_Specificity": result_test["Specificity"],
        "Test_BalAcc": result_test["Balanced Accuracy"],
        "Test_Kappa": result_test["Kappa"],
        "Test_Matthews": result_test["Matthews"],
        "Test_AUROC": result_test["AUROC"],
        "Test_Accuracy_bestCutoff": result_test_best["Accuracy"],
        "Test_Sensitivity_bestCutoff": result_test_best["Sensitivity"],
        "Test_Specificity_bestCutoff": result_test_best["Specificity"],
        "Test_BalAcc_bestCutoff": result_test_best["Balanced Accuracy"],
        "Test_Kappa_bestCutoff": result_test_best["Kappa"],
        "Test_Matthews_bestCutoff": result_test_best["Matthews"],
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
    local_ckpt_path = Path(local_ckpt_path).expanduser().resolve()
    if not local_ckpt_path.is_file():
        raise FileNotFoundError(f"Local checkpoint not found: {local_ckpt_path}")

    local_tabicl_summary = run_tabicl_eval(
        model_label=args.local_model_label,
        model_path=str(local_ckpt_path),
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
