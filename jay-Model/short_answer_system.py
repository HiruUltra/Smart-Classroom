# short_answer_system.py
# ==========================================================
# Keyword-based short answer grading + ML trainer
#
# RUN DIRECTLY:
#   python short_answer_system.py
#
# It will train from your question bank:
#   it_short_answer_dataset.csv
# (auto-detects /mnt/data/it_short_answer_dataset.csv if needed)
#
# Saves outputs:
#   learning_curve_classification.png
#   learning_curve_regression.png
#   precision_recall_macro.png
#   residuals_regression.png
#   confusion_matrix.png
#   classification_report.txt / .csv
#   accuracy_table.csv / .png
#   model_leaderboard.csv
#   regression_leaderboard.csv
#   metrics.json
#   regression_metrics.json
#   short_answer_grader_classifier.joblib
#   short_answer_grader_regressor.joblib
# ==========================================================

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    KFold,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC, SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

RANDOM_STATE = 42


# ---------------- BASIC TEXT HELPERS ----------------

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_keywords(kw_str: str) -> List[str]:
    if not isinstance(kw_str, str):
        return []
    parts = [k.strip().lower() for k in kw_str.split("|")]
    return [p for p in parts if p]


def keyword_match_stats(answer_text: str, main_keywords: List[str], opt_keywords: List[str]) -> Dict:
    ans_clean = clean_text(answer_text)

    def count_matches(keywords: List[str]) -> Tuple[int, List[str]]:
        matched = []
        for kw in keywords:
            kw_clean = clean_text(kw)
            if kw_clean and kw_clean in ans_clean:
                matched.append(kw)
        return len(matched), matched

    main_count, main_matched = count_matches(main_keywords)
    opt_count, opt_matched = count_matches(opt_keywords)

    total_main = len(main_keywords)
    main_coverage = main_count / total_main if total_main > 0 else 0.0

    return {
        "main_count": main_count,
        "opt_count": opt_count,
        "main_matched": main_matched,
        "opt_matched": opt_matched,
        "main_coverage": main_coverage,
        "total_main": total_main,
    }


def grade_by_keywords(stats: Dict) -> Dict:
    cov = stats["main_coverage"]
    main_count = stats["main_count"]

    if cov >= 0.75 and main_count >= 3:
        level = "GOOD"
        marks = 10
    elif cov >= 0.5:
        level = "PARTIAL"
        marks = 5
    elif main_count >= 1:
        level = "WEAK"
        marks = 2
    else:
        level = "INCORRECT"
        marks = 0

    return {"level": level, "marks": marks}


# ---------------- SYNTHETIC DATA GENERATION ----------------

def _shuffle_words(text: str, p: float = 0.15, rng: np.random.Generator = None) -> str:
    rng = rng or np.random.default_rng(RANDOM_STATE)
    words = clean_text(text).split()
    if len(words) < 6:
        return " ".join(words)
    keep = [w for w in words if rng.random() > p]
    if len(keep) < max(3, len(words) // 3):
        keep = words[:]
    if rng.random() < 0.4:
        rng.shuffle(keep)
    return " ".join(keep)


def _build_answer_from_keywords(keywords: List[str], take_n: int, rng: np.random.Generator) -> str:
    if not keywords:
        return ""
    kws = keywords[:]
    rng.shuffle(kws)
    picked = kws[:max(1, min(take_n, len(kws)))]
    return " ".join([clean_text(k) for k in picked]).strip()


def build_synthetic_labelled_dataset(
    question_bank_csv: str,
    out_csv: Optional[str] = None,
    neg_per_question: int = 2,
    rng_seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Creates labelled data (student_answer, grade_label, marks) from question bank.

    Required columns in question bank:
      - question_text
      - ideal_answer
      - keywords_main
    Optional:
      - keywords_optional
    """
    rng = np.random.default_rng(rng_seed)
    qdf = pd.read_csv(question_bank_csv, encoding="latin-1")

    required = ["question_text", "ideal_answer", "keywords_main"]
    for c in required:
        if c not in qdf.columns:
            raise ValueError(f"Question bank missing required column: {c}")

    ideal_pool = qdf["ideal_answer"].astype(str).tolist()

    rows = []
    for _, row in qdf.iterrows():
        q = str(row["question_text"])
        ideal = str(row["ideal_answer"])
        main_kws = parse_keywords(row.get("keywords_main", ""))
        opt_kws = parse_keywords(row.get("keywords_optional", ""))

        # GOOD
        good_ans = ideal if rng.random() < 0.5 else _shuffle_words(ideal, p=0.12, rng=rng)
        rows.append({"question_text": q, "student_answer": good_ans, "grade_label": "GOOD", "marks": 10})

        # PARTIAL
        take = max(2, int(round(len(main_kws) * 0.5))) if main_kws else 2
        partial_ans = _build_answer_from_keywords(main_kws, take_n=take, rng=rng)
        if opt_kws and rng.random() < 0.5:
            partial_ans = (partial_ans + " " + _build_answer_from_keywords(opt_kws, take_n=1, rng=rng)).strip()
        rows.append({
            "question_text": q,
            "student_answer": partial_ans if partial_ans else _shuffle_words(ideal, p=0.45, rng=rng),
            "grade_label": "PARTIAL",
            "marks": 5,
        })

        # WEAK
        weak_ans = _build_answer_from_keywords(main_kws, take_n=1, rng=rng)
        if not weak_ans:
            weak_ans = "not sure"
        rows.append({"question_text": q, "student_answer": weak_ans, "grade_label": "WEAK", "marks": 2})

        # INCORRECT (wrong context answers)
        for _k in range(neg_per_question):
            j = int(rng.integers(0, len(ideal_pool)))
            wrong_ans = str(ideal_pool[j])
            if wrong_ans.strip() == ideal.strip():
                wrong_ans = "i don't know"
            if rng.random() < 0.25:
                wrong_ans = "i don't know"
            rows.append({"question_text": q, "student_answer": wrong_ans, "grade_label": "INCORRECT", "marks": 0})

    df = pd.DataFrame(rows)
    df["question_text"] = df["question_text"].astype(str).apply(clean_text)
    df["student_answer"] = df["student_answer"].astype(str).apply(clean_text)
    df["combined_text"] = df["question_text"] + " [SEP] " + df["student_answer"]

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
    return df


# ---------------- PLOTTING HELPERS ----------------

def _save_text_table_as_image(df: pd.DataFrame, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.35 * len(df)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_learning_curve_classifier(pipe, X, y, out_path: str, cv):
    sizes, train_scores, val_scores = learning_curve(
        pipe,
        X,
        y,
        cv=cv,
        scoring="f1_weighted",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1,
    )
    fig, ax = plt.subplots()
    ax.plot(sizes, train_scores.mean(axis=1), lw=2, label="train f1_weighted")
    ax.plot(sizes, val_scores.mean(axis=1), lw=2, label="cv f1_weighted")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve (Classification)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_learning_curve_regressor(pipe, X, y, out_path: str, cv):
    sizes, train_scores, val_scores = learning_curve(
        pipe,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1,
    )
    train_mae = -train_scores.mean(axis=1)
    val_mae = -val_scores.mean(axis=1)

    fig, ax = plt.subplots()
    ax.plot(sizes, train_mae, lw=2, label="train MAE")
    ax.plot(sizes, val_mae, lw=2, label="cv MAE")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("Learning Curve (Regression)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_precision_recall_macro(pipe, X_test, y_test, classes, out_path: str):
    prob = pipe.predict_proba(X_test)

    y_bin = np.zeros((len(y_test), len(classes)), dtype=int)
    class_to_i = {c: i for i, c in enumerate(classes)}
    for r, lbl in enumerate(y_test):
        y_bin[r, class_to_i[lbl]] = 1

    fig, ax = plt.subplots()
    ap_scores = []
    for i, c in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], prob[:, i])
        ap = average_precision_score(y_bin[:, i], prob[:, i])
        ap_scores.append(ap)
        ax.plot(recall, precision, lw=1.8, label=f"{c} (AP={ap:.3f})")

    macro_ap = float(np.mean(ap_scores)) if ap_scores else 0.0
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall (Macro AP={macro_ap:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_residuals(y_true, y_pred, out_path: str):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.plot(y_pred, residuals, "o", markersize=3)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (true - pred)")
    ax.set_title("Residuals (Regression)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------- MAIN TRAINER ----------------

def train_ml_grader(
    labelled_csv: Optional[str] = None,
    question_bank_csv: Optional[str] = None,
    text_col: str = "student_answer",
    label_col: str = "grade_label",
    output_dir: str = None,
    build_synthetic_if_needed: bool = True,
):
    """
    Option A (labelled):
      labelled_csv columns: question_text, student_answer, grade_label

    Option B (no labels):
      question_bank_csv columns: question_text, ideal_answer, keywords_main, keywords_optional
      -> generates synthetic labelled dataset automatically.
    """
    import joblib

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs_short_answer", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # -------- Load dataset --------
    if labelled_csv:
        df = pd.read_csv(labelled_csv, encoding="latin-1")
        if "question_text" not in df.columns or text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Labelled CSV must contain: question_text, {text_col}, {label_col}")
        df["question_text"] = df["question_text"].astype(str).apply(clean_text)
        df[text_col] = df[text_col].astype(str).apply(clean_text)
        df["combined_text"] = df["question_text"] + " [SEP] " + df[text_col]
        df["marks"] = df.get("marks", np.nan)
    else:
        if not question_bank_csv:
            raise ValueError("Provide either labelled_csv OR question_bank_csv.")
        if not build_synthetic_if_needed:
            raise ValueError("No labelled_csv provided and build_synthetic_if_needed=False.")
        synthetic_path = os.path.join(output_dir, "synthetic_training_data.csv")
        df = build_synthetic_labelled_dataset(question_bank_csv, out_csv=synthetic_path)
        print("Saved synthetic labelled dataset to:", synthetic_path)

    X_text = df["combined_text"]
    y = df[label_col].astype(str)

    label_counts = y.value_counts()
    print("Label counts:", label_counts.to_dict())

    # ---------------- CLASSIFICATION MODEL ZOO ----------------
    svc_cal = CalibratedClassifierCV(
        estimator=LinearSVC(),
        method="sigmoid",
        cv=3,
    )

    model_defs = {
        "logreg": LogisticRegression(max_iter=1000, n_jobs=-1),
        "svc_calibrated": svc_cal,
        "rf": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }

    min_class_count = int(label_counts.min())
    possible_splits = min(5, min_class_count)
    if possible_splits < 2:
        possible_splits = 2

    skf = StratifiedKFold(n_splits=possible_splits, shuffle=True, random_state=RANDOM_STATE)

    results_rows = []
    for name, clf in model_defs.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)),
            ("clf", clf),
        ])
        acc_scores = cross_val_score(pipe, X_text, y, cv=skf, scoring="accuracy", n_jobs=-1)
        f1_scores = cross_val_score(pipe, X_text, y, cv=skf, scoring="f1_weighted", n_jobs=-1)

        row = {
            "model": name,
            "cv_accuracy_mean": float(acc_scores.mean()),
            "cv_accuracy_std": float(acc_scores.std()),
            "cv_f1_weighted_mean": float(f1_scores.mean()),
            "cv_f1_weighted_std": float(f1_scores.std()),
        }
        results_rows.append(row)
        print(
            f"[{name}] acc={row['cv_accuracy_mean']:.3f} ± {row['cv_accuracy_std']:.3f}, "
            f"f1={row['cv_f1_weighted_mean']:.3f} ± {row['cv_f1_weighted_std']:.3f}"
        )

    results_df = pd.DataFrame(results_rows).sort_values("cv_f1_weighted_mean", ascending=False)
    leaderboard_path = os.path.join(output_dir, "model_leaderboard.csv")
    results_df.to_csv(leaderboard_path, index=False)
    print("Saved model leaderboard to:", leaderboard_path)

    best_name = results_df.iloc[0]["model"]
    best_clf = model_defs[best_name]
    print("Best classifier:", best_name)

    # ---------------- TRAIN/TEST SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    best_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)),
        ("clf", best_clf),
    ])

    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    # ---------------- METRICS + REPORTS ----------------
    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")

    metrics = {
        "best_model": best_name,
        "test_accuracy": float(acc),
        "test_f1_weighted": float(f1w),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "label_counts": label_counts.to_dict(),
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to:", metrics_path)

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    rep_txt_path = os.path.join(output_dir, "classification_report.txt")
    with open(rep_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Saved classification report to:", rep_txt_path)

    rep_csv_path = os.path.join(output_dir, "classification_report.csv")
    pd.DataFrame(report_dict).transpose().to_csv(rep_csv_path)
    print("Saved classification report CSV to:", rep_csv_path)

    # accuracy table
    acc_table = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
    acc_table_path = os.path.join(output_dir, "accuracy_table.csv")
    acc_table.to_csv(acc_table_path, index=False)
    print("Saved accuracy table CSV to:", acc_table_path)

    acc_table_img = os.path.join(output_dir, "accuracy_table.png")
    _save_text_table_as_image(acc_table.round(3), acc_table_img, title="Accuracy / Precision / Recall / F1 Table")
    print("Saved accuracy table image to:", acc_table_img)

    # confusion matrix
    classes = sorted(list(np.unique(y)))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax_cm, cmap="Blues", values_format="d", colorbar=False)
    plt.title(f"Confusion Matrix - {best_name}")
    fig_cm.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=200)
    plt.close(fig_cm)
    print("Saved confusion matrix to:", cm_path)

    # precision-recall macro
    pr_path = os.path.join(output_dir, "precision_recall_macro.png")
    try:
        _plot_precision_recall_macro(best_pipe, X_test, list(y_test), classes, pr_path)
        print("Saved precision-recall macro to:", pr_path)
    except Exception as e:
        print("Skipping precision-recall macro (model may not support predict_proba):", str(e))

    # learning curve classification
    lc_cls_path = os.path.join(output_dir, "learning_curve_classification.png")
    _plot_learning_curve_classifier(best_pipe, X_text, y, lc_cls_path, cv=skf)
    print("Saved learning curve (classification) to:", lc_cls_path)

    # save classifier
    clf_model_path = os.path.join(output_dir, "short_answer_grader_classifier.joblib")
    import joblib
    joblib.dump(best_pipe, clf_model_path)
    print("Saved trained classifier to:", clf_model_path)

    # ---------------- REGRESSION (MARKS) ----------------
    marks_map = {"GOOD": 10, "PARTIAL": 5, "WEAK": 2, "INCORRECT": 0}
    y_marks = df.get("marks")
    if y_marks is None or pd.isna(y_marks).all():
        y_marks = y.map(marks_map).astype(float)
    else:
        y_marks = pd.to_numeric(y_marks, errors="coerce").fillna(y.map(marks_map)).astype(float)

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_text, y_marks, test_size=0.2, random_state=RANDOM_STATE
    )

    reg_defs = {
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "svr": SVR(kernel="rbf", C=10.0, epsilon=0.2),
        "rf_reg": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    reg_rows = []
    for name, reg in reg_defs.items():
        pipe_r = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)),
            ("reg", reg),
        ])
        scores = cross_val_score(pipe_r, X_text, y_marks, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
        reg_rows.append({
            "model": name,
            "cv_mae_mean": float((-scores).mean()),
            "cv_mae_std": float((-scores).std()),
        })

    reg_df = pd.DataFrame(reg_rows).sort_values("cv_mae_mean", ascending=True)
    reg_leader_path = os.path.join(output_dir, "regression_leaderboard.csv")
    reg_df.to_csv(reg_leader_path, index=False)
    print("Saved regression leaderboard to:", reg_leader_path)

    best_reg_name = reg_df.iloc[0]["model"]
    best_reg = reg_defs[best_reg_name]
    print("Best regressor:", best_reg_name)

    reg_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)),
        ("reg", best_reg),
    ])
    reg_pipe.fit(Xr_train, yr_train)
    yr_pred = reg_pipe.predict(Xr_test)

    mae = mean_absolute_error(yr_test, yr_pred)
    rmse = float(np.sqrt(mean_squared_error(yr_test, yr_pred)))
    r2 = r2_score(yr_test, yr_pred)

    reg_metrics = {
        "best_regressor": best_reg_name,
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2),
        "n_train": int(len(Xr_train)),
        "n_test": int(len(Xr_test)),
    }
    reg_metrics_path = os.path.join(output_dir, "regression_metrics.json")
    with open(reg_metrics_path, "w", encoding="utf-8") as f:
        json.dump(reg_metrics, f, indent=2)
    print("Saved regression metrics to:", reg_metrics_path)

    # learning curve regression
    lc_reg_path = os.path.join(output_dir, "learning_curve_regression.png")
    _plot_learning_curve_regressor(reg_pipe, X_text, y_marks, lc_reg_path, cv=kf)
    print("Saved learning curve (regression) to:", lc_reg_path)

    # residuals regression
    residuals_path = os.path.join(output_dir, "residuals_regression.png")
    _plot_residuals(yr_test.to_numpy(), yr_pred, residuals_path)
    print("Saved residuals plot to:", residuals_path)

    # save regressor
    reg_model_path = os.path.join(output_dir, "short_answer_grader_regressor.joblib")
    joblib.dump(reg_pipe, reg_model_path)
    print("Saved trained regressor to:", reg_model_path)

    print("\n✅ All outputs saved in:", output_dir)


# ---------------- RUN DIRECTLY ----------------

def _resolve_question_bank_path(default_name: str = "it_short_answer_dataset.csv") -> str:
    """
    Auto-detect dataset path:
      1) current folder: it_short_answer_dataset.csv
      2) /mnt/data/it_short_answer_dataset.csv (common in notebooks)
    """
    if os.path.isfile(default_name):
        return default_name
    alt = os.path.join("/mnt/data", default_name)
    if os.path.isfile(alt):
        return alt
    return default_name  # fallback (will raise clear error later)


if __name__ == "__main__":
    qb_path = _resolve_question_bank_path("it_short_answer_dataset.csv")
    print("Using question bank:", qb_path)

    train_ml_grader(
        labelled_csv=None,
        question_bank_csv=qb_path,
        output_dir=None,
        build_synthetic_if_needed=True,
    )
