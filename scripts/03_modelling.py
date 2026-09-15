"""
03 - Model training & comparison for TB mortality prediction.

Six algorithms (LR, RF, XGBoost, GradientBoosting, SVM, KNN) in an imblearn
pipeline:  ColumnTransformer (scale numeric + one-hot categorical) -> SMOTE
(train folds only) -> classifier. 80/20 stratified hold-out + 5-fold stratified
CV on the training set. Single-threaded (n_jobs=1) for macOS OpenMP safety.

Outputs:
  results/model_comparison_cv.csv, model_comparison_test.csv
  results/bootstrap_ci_best.csv, calibration_summary_best.csv, threshold_metrics_best.csv
  results/decision_curve_best.csv, sensitivity_model_comparison.csv
  results/confusion_matrix_best.csv, classification_report_best.csv
  figures/fig_06_model_comparison.png, fig_07_roc_curves.png, fig_08_confusion_matrix_best.png
  figures/fig_13_calibration_best.png, fig_14_decision_curve_best.png
  models/best_model.pkl  (+ pipeline_<model>.pkl for each)
"""
import config  # sets thread env vars first
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, roc_auc_score,
                             recall_score, precision_score, confusion_matrix,
                             classification_report, roc_curve, make_scorer)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", context="paper")
import plotstyle
plotstyle.use_style()
plt.rcParams.update({"savefig.dpi": config.PLOT_DPI, "savefig.bbox": "tight",
                     "axes.titleweight": "bold", "font.size": 10})
RS = config.RANDOM_STATE

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
df = config.load_processed()
X = df[config.FEATURES].copy()
y = df[config.TARGET].astype(int).values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RS)
print(f"Train {X_tr.shape[0]} (died {int(y_tr.sum())}) | Test {X_te.shape[0]} (died {int(y_te.sum())})")

preprocess = ColumnTransformer([
    ("num", StandardScaler(), config.NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
])


def make_pipe(clf):
    return ImbPipeline([("pre", preprocess),
                        ("smote", SMOTE(random_state=RS)),
                        ("clf", clf)])


def make_no_smote_pipe(clf):
    return ImbPipeline([("pre", preprocess), ("clf", clf)])


def metric_row(label, y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": label,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1),
        "specificity": recall_score(y_true, y_pred, pos_label=0),
        "ppv": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
    }


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RS, n_jobs=1),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=1),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9, random_state=RS,
                             n_jobs=1, nthread=1, eval_metric="logloss",
                             use_label_encoder=False, verbosity=0),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
    "SVM": SVC(kernel="rbf", probability=True, random_state=RS),
    "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=1),
}

# ----------------------------------------------------------------------------
# 5-fold stratified CV on the training set
# ----------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)


def auc_scorer(estimator, X, y_true):
    # Call predict_proba directly: avoids the sklearn 1.6 / xgboost 1.5
    # is_classifier tag mismatch that breaks the built-in 'roc_auc' string scorer.
    return roc_auc_score(y_true, estimator.predict_proba(X)[:, 1])


scoring = {"accuracy": "accuracy", "f1_weighted": "f1_weighted", "roc_auc": auc_scorer,
           "sensitivity": make_scorer(recall_score, pos_label=1),
           "specificity": make_scorer(recall_score, pos_label=0)}

cv_rows, test_rows, fitted, roc_data = [], [], {}, {}
for name, clf in MODELS.items():
    pipe = make_pipe(clf)
    cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
    cv_rows.append({"model": name, **{
        f"cv_{m}": f"{cvres['test_'+m].mean():.3f} ± {cvres['test_'+m].std():.3f}"
        for m in scoring}})

    # Fit on full training set, evaluate on held-out test
    pipe.fit(X_tr, y_tr)
    fitted[name] = pipe
    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    test_rows.append(metric_row(name, y_te, y_pred, y_proba))
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    roc_data[name] = (fpr, tpr, roc_auc_score(y_te, y_proba))
    cvm = cvres["test_roc_auc"].mean()
    print(f"  {name:<22} CV AUC={cvm:.3f}  Test AUC={roc_data[name][2]:.3f}  "
          f"Sens={test_rows[-1]['sensitivity']:.3f}  Spec={test_rows[-1]['specificity']:.3f}")
    joblib.dump(pipe, config.MODELS_DIR / f"pipeline_{name.replace(' ', '_')}.pkl")

cv_df = pd.DataFrame(cv_rows)
test_df = pd.DataFrame(test_rows).round(3)
cv_df.to_csv(config.RESULTS_DIR / "model_comparison_cv.csv", index=False)
test_df.to_csv(config.RESULTS_DIR / "model_comparison_test.csv", index=False)

# ----------------------------------------------------------------------------
# Select best model by mean CV ROC-AUC
# ----------------------------------------------------------------------------
cv_auc = {r["model"]: float(r["cv_roc_auc"].split(" ")[0]) for r in cv_rows}
best = max(cv_auc, key=cv_auc.get)
best_pipe = fitted[best]
joblib.dump(best_pipe, config.MODELS_DIR / "best_model.pkl")
print(f"\nBEST MODEL (by CV ROC-AUC): {best}")
print("\n--- Test metrics ---")
print(test_df.to_string(index=False))

# Confusion matrix + classification report for best model
y_pred_best = best_pipe.predict(X_te)
cm = confusion_matrix(y_te, y_pred_best)
pd.DataFrame(cm, index=["Actual Non-death", "Actual Died"],
             columns=["Pred Non-death", "Pred Died"]).to_csv(
    config.RESULTS_DIR / "confusion_matrix_best.csv")
rep = classification_report(y_te, y_pred_best, target_names=["Non-death", "Died"],
                            output_dict=True, zero_division=0)
pd.DataFrame(rep).transpose().round(3).to_csv(config.RESULTS_DIR / "classification_report_best.csv")

with open(config.RESULTS_DIR / "best_model.json", "w") as f:
    json.dump({"best_model": best, "selection_metric": "mean CV ROC-AUC",
               "cv_roc_auc": cv_auc[best],
               "test_metrics": test_df[test_df.model == best].iloc[0].to_dict()}, f, indent=2)

# ----------------------------------------------------------------------------
# Q1 journal audit additions: uncertainty, calibration, threshold utility,
# decision curves, and sensitivity analysis without SMOTE.
# ----------------------------------------------------------------------------
y_proba_best = best_pipe.predict_proba(X_te)[:, 1]


def metrics_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "classified_high_risk": int(tp + fp),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


boot_rng = np.random.default_rng(RS)
boot_rows = []
for _ in range(2000):
    idx = boot_rng.integers(0, len(y_te), len(y_te))
    if len(np.unique(y_te[idx])) < 2:
        continue
    p_b = y_proba_best[idx]
    y_b = y_te[idx]
    pred_b = (p_b >= 0.5).astype(int)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_b, pred_b).ravel()
    boot_rows.append({
        "accuracy": accuracy_score(y_b, pred_b),
        "f1_weighted": f1_score(y_b, pred_b, average="weighted"),
        "roc_auc": roc_auc_score(y_b, p_b),
        "sensitivity": recall_score(y_b, pred_b, pos_label=1),
        "specificity": recall_score(y_b, pred_b, pos_label=0),
        "ppv": precision_score(y_b, pred_b, pos_label=1, zero_division=0),
        "npv": tn_b / (tn_b + fn_b) if (tn_b + fn_b) else np.nan,
    })
boot_df = pd.DataFrame(boot_rows)
ci_rows = []
point = metric_row(best, y_te, y_pred_best, y_proba_best)
for metric in ["accuracy", "f1_weighted", "roc_auc", "sensitivity", "specificity", "ppv", "npv"]:
    ci_rows.append({
        "metric": metric,
        "estimate": point[metric],
        "ci_lower": boot_df[metric].quantile(0.025),
        "ci_upper": boot_df[metric].quantile(0.975),
    })
pd.DataFrame(ci_rows).round(3).to_csv(config.RESULTS_DIR / "bootstrap_ci_best.csv", index=False)

prob_true, prob_pred = calibration_curve(y_te, y_proba_best, n_bins=8, strategy="quantile")
pd.DataFrame({"mean_predicted_risk": prob_pred, "observed_risk": prob_true}).round(4).to_csv(
    config.RESULTS_DIR / "calibration_curve_best.csv", index=False)
logit_p = np.log(np.clip(y_proba_best, 1e-6, 1 - 1e-6) / np.clip(1 - y_proba_best, 1e-6, 1 - 1e-6))
try:
    import statsmodels.api as sm
    cal_fit = sm.Logit(y_te, sm.add_constant(logit_p)).fit(disp=False)
    cal_intercept = float(cal_fit.params[0])
    cal_slope = float(cal_fit.params[1])
except Exception:
    cal_intercept = np.nan
    cal_slope = np.nan
cal_summary = pd.DataFrame([{
    "model": best,
    "brier_score": brier_score_loss(y_te, y_proba_best),
    "calibration_intercept": cal_intercept,
    "calibration_slope": cal_slope,
    "mean_predicted_risk": float(np.mean(y_proba_best)),
    "observed_risk": float(np.mean(y_te)),
}])
cal_summary.round(3).to_csv(config.RESULTS_DIR / "calibration_summary_best.csv", index=False)

fig, ax = plt.subplots(figsize=(5.8, 5.2))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal calibration")
ax.plot(prob_pred, prob_true, marker="o", lw=2, label=best)
ax.set_xlabel("Mean predicted risk")
ax.set_ylabel("Observed risk")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="upper left")
fig.savefig(config.FIGURES_DIR / "fig_13_calibration_best.png")
plt.close(fig)

thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
pd.DataFrame([metrics_at_threshold(y_te, y_proba_best, t) for t in thresholds]).round(3).to_csv(
    config.RESULTS_DIR / "threshold_metrics_best.csv", index=False)

pt_grid = np.round(np.arange(0.05, 0.81, 0.01), 2)
n = len(y_te)
prevalence = float(np.mean(y_te))
dca_rows = []
for pt in pt_grid:
    row = metrics_at_threshold(y_te, y_proba_best, pt)
    net_benefit_model = (row["TP"] / n) - (row["FP"] / n) * (pt / (1 - pt))
    net_benefit_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
    dca_rows.append({
        "threshold": pt,
        "net_benefit_model": net_benefit_model,
        "net_benefit_treat_all": net_benefit_all,
        "net_benefit_treat_none": 0.0,
    })
dca_df = pd.DataFrame(dca_rows)
dca_df.round(4).to_csv(config.RESULTS_DIR / "decision_curve_best.csv", index=False)

fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.plot(dca_df["threshold"], dca_df["net_benefit_model"], lw=2.4, label=best)
ax.plot(dca_df["threshold"], dca_df["net_benefit_treat_all"], lw=1.5, ls="--", label="Treat all")
ax.axhline(0, color="black", lw=1, ls=":", label="Treat none")
ax.set_xlabel("Risk threshold")
ax.set_ylabel("Net benefit")
ax.set_xlim(0.05, 0.80)
ax.legend(frameon=False, loc="upper right")
fig.savefig(config.FIGURES_DIR / "fig_14_decision_curve_best.png")
plt.close(fig)

sensitivity_variants = {
    "Primary logistic regression + SMOTE": make_pipe(LogisticRegression(max_iter=2000, random_state=RS, n_jobs=1)),
    "Logistic regression without SMOTE": make_no_smote_pipe(LogisticRegression(max_iter=2000, random_state=RS, n_jobs=1)),
    "Logistic regression with class weighting": make_no_smote_pipe(
        LogisticRegression(max_iter=2000, random_state=RS, n_jobs=1, class_weight="balanced")),
}
sens_rows = []
for label, pipe in sensitivity_variants.items():
    cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    row = metric_row(label, y_te, pred, proba)
    row["cv_roc_auc"] = cvres["test_roc_auc"].mean()
    row["cv_roc_auc_sd"] = cvres["test_roc_auc"].std()
    sens_rows.append(row)
pd.DataFrame(sens_rows).round(3).to_csv(config.RESULTS_DIR / "sensitivity_model_comparison.csv", index=False)

# ----------------------------------------------------------------------------
# Figure 6 - model comparison (grouped bars, test metrics)
# ----------------------------------------------------------------------------
metrics = ["accuracy", "f1_weighted", "roc_auc", "sensitivity", "specificity"]
labels = ["Accuracy", "Weighted F1", "ROC-AUC", "Sensitivity", "Specificity"]
pal = sns.color_palette("Set2", len(metrics))
xpos = np.arange(len(MODELS)); w = 0.16
fig, ax = plt.subplots(figsize=(12, 5.5))
for i, (m, lab) in enumerate(zip(metrics, labels)):
    vals = test_df.set_index("model").loc[list(MODELS), m].values
    bars = ax.bar(xpos + (i - 2) * w, vals, w, label=lab, color=pal[i],
                  edgecolor="black", linewidth=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.5, rotation=90)
ax.set_xticks(xpos); ax.set_xticklabels(list(MODELS), rotation=15, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
# figure-level title removed (caption supplied outside the plot)
ax.legend(ncol=5, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28))
fig.savefig(config.FIGURES_DIR / "fig_06_model_comparison.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 7 - ROC curves
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
for name, (fpr, tpr, auc) in sorted(roc_data.items(), key=lambda x: -x[1][2]):
    lw = 2.5 if name == best else 1.3
    ax.plot(fpr, tpr, lw=lw, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
ax.set_xlabel("1 - Specificity (FPR)"); ax.set_ylabel("Sensitivity (TPR)")
# figure-level title removed (caption supplied outside the plot)
ax.legend(frameon=True, fontsize=8, loc="lower right")
fig.savefig(config.FIGURES_DIR / "fig_07_roc_curves.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 8 - confusion matrix (best model)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4.3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False,
            xticklabels=["Non-death", "Died"], yticklabels=["Non-death", "Died"], ax=ax,
            annot_kws={"fontsize": 14, "fontweight": "bold"})
for text, value in zip(ax.texts, cm.flatten()):
    text.set_color("white" if value > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
# figure-level title removed (caption supplied outside the plot)
fig.savefig(config.FIGURES_DIR / "fig_08_confusion_matrix_best.png")
plt.close(fig)

print(f"\nSaved CV/test CSVs, 3 figures, and {len(MODELS)+1} model pkl files.")
