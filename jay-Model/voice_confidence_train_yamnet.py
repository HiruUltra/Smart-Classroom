#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
voice_confidence_train_yamnet.py
============================================
Upgrade Path (Recommended): Pretrained embeddings + small ML classifier

- Uses YAMNet (TF Hub) to extract 1024-d embeddings per frame
- Pools embeddings to a fixed-length vector (mean or mean+std)
- Trains a lightweight classifier (default: LogisticRegression)
- Saves:
    * voice_confidence_model.joblib (root + outputs folder)
    * classification_report.csv + classification_report.png
    * confusion_matrix.png
    * learning_curve_classification.csv + learning_curve_classification.png
    * precision_recall_macro.csv + precision_recall_macro.png
    * accuracy_table.csv + accuracy_table.png
    * y_true_y_pred.csv
    * (requested) learning_curve_regression.png + residuals_regression.png (N/A placeholders)

Folder structure:
  training_data/
      Confident/*.wav
      Hesitant/*.wav
      Nervous/*.wav

First run will download YAMNet from TF Hub.
"""

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# Audio
import librosa

# ML
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
)

# Plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Deep embeddings (YAMNet)
import tensorflow as tf
import tensorflow_hub as hub


# ---------------- CONFIG ----------------
DATA_DIR = "training_data"
SAMPLE_RATE = 16000

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

# Pooling for per-frame embeddings -> fixed vector
# "mean"      => 1024 dims
# "mean_std"  => 2048 dims (often better)
EMBED_POOL = "mean_std"

# Classifier choice (keep it simple + strong)
# For PR curves we want predict_proba => LogisticRegression works well.
CLF_NAME = "logreg"  # (kept for future extension)

# Output
OUT_ROOT = "outputs_voice"
ROOT_MODEL_PATH = "voice_confidence_model.joblib"


# ---------------- EMBEDDING (YAMNet) ----------------
_yamnet_model = None

def get_yamnet():
    global _yamnet_model
    if _yamnet_model is None:
        print(f"[INFO] Loading YAMNet from TF Hub: {YAMNET_HANDLE}")
        _yamnet_model = hub.load(YAMNET_HANDLE)
    return _yamnet_model


def extract_yamnet_embedding(file_path: str) -> np.ndarray | None:
    """
    Loads audio -> waveform float32 mono 16k
    Runs YAMNet -> embeddings (frames, 1024)
    Pools -> fixed vector
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"[WARN] Failed to load {file_path}: {e}")
        return None

    if y.size == 0:
        print(f"[WARN] Empty audio: {file_path}")
        return None

    # YAMNet expects float32 waveform in [-1, 1]
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)

    yamnet = get_yamnet()
    scores, embeddings, spectrogram = yamnet(waveform)  # embeddings: (N, 1024)

    emb = embeddings.numpy()
    if emb.size == 0:
        print(f"[WARN] No embeddings produced: {file_path}")
        return None

    if EMBED_POOL == "mean":
        vec = emb.mean(axis=0)
    elif EMBED_POOL == "mean_std":
        vec = np.concatenate([emb.mean(axis=0), emb.std(axis=0)], axis=0)
    else:
        raise ValueError("EMBED_POOL must be 'mean' or 'mean_std'")

    return vec.astype(np.float32)


def load_dataset(data_dir: str):
    """
    Walk training_data/<label>/*.(wav|mp3|m4a) and build X, y.
    Each subfolder name is the class label.
    """
    X, y = [], []

    classes = []
    for name in os.listdir(data_dir):
        full = os.path.join(data_dir, name)
        if os.path.isdir(full):
            classes.append(name)

    if not classes:
        raise RuntimeError(
            f"No class folders found in {data_dir}. Expected e.g. 'Confident', 'Hesitant', 'Nervous'."
        )

    print("[INFO] Found classes:", classes)

    for label in classes:
        folder = os.path.join(data_dir, label)
        pattern_list = [
            os.path.join(folder, "*.wav"),
            os.path.join(folder, "*.mp3"),
            os.path.join(folder, "*.m4a"),
        ]

        files = []
        for pattern in pattern_list:
            files.extend(glob.glob(pattern))

        if not files:
            print(f"[WARN] No audio files found in {folder}")
            continue

        print(f"[INFO] Loading {len(files)} files for class '{label}'")

        for fpath in files:
            vec = extract_yamnet_embedding(fpath)
            if vec is None:
                continue
            X.append(vec)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    if X.size == 0:
        raise RuntimeError("No embeddings extracted. Check audio files and paths.")

    return X, y


# ---------------- PLOTS / HELPERS ----------------
def save_table_image(df: pd.DataFrame, title: str, out_path: str, font_size: int = 9):
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(df.columns)), max(2.5, 0.35 * len(df) + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=np.round(df.values, 4),
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.2, 1.25)
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_na_placeholder(title: str, out_path: str, note: str):
    plt.figure(figsize=(7, 2.5))
    plt.axis("off")
    plt.title(title)
    plt.text(0.02, 0.5, note, fontsize=11)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_learning_curve_classification(estimator, X, y, cv, out_csv, out_png):
    train_sizes = np.linspace(0.1, 1.0, 5)
    sizes, train_scores, val_scores = learning_curve(
        estimator,
        X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    lc_df = pd.DataFrame({
        "train_size": sizes,
        "train_acc_mean": train_mean,
        "train_acc_std": train_std,
        "val_acc_mean": val_mean,
        "val_acc_std": val_std,
    })
    lc_df.to_csv(out_csv, index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, train_mean, marker="o", label="Train Accuracy")
    plt.plot(sizes, val_mean, marker="o", label="CV Accuracy")
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.ylim(0, 1)
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve (Classification)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_precision_recall_macro(y_true, y_proba, class_names, out_csv, out_png):
    """
    Macro-averaged PR curve by interpolating per-class PR onto a shared recall grid.
    Also saves per-class average precision and macro AP in CSV.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))

    ap_per_class = []
    curves = []

    recall_grid = np.linspace(0.0, 1.0, 250)
    prec_interp_all = []

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        ap_per_class.append(ap)

        # Ensure recall is increasing for interpolation
        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]

        # Interpolate precision at common recall grid
        prec_interp = np.interp(recall_grid, recall_sorted, precision_sorted, left=precision_sorted[0], right=precision_sorted[-1])
        prec_interp_all.append(prec_interp)

        curves.append((i, recall, precision))

    prec_macro = np.mean(np.vstack(prec_interp_all), axis=0)
    ap_macro = average_precision_score(y_bin, y_proba, average="macro")

    # Save table
    pr_df = pd.DataFrame({
        "class": list(class_names) + ["MACRO"],
        "average_precision": list(ap_per_class) + [ap_macro]
    })
    pr_df.to_csv(out_csv, index=False)

    # Plot
    plt.figure(figsize=(7, 5))
    for i, recall, precision in curves:
        plt.plot(recall, precision, linewidth=1, label=f"{class_names[i]} (AP={ap_per_class[i]:.3f})")

    plt.plot(recall_grid, prec_macro, linewidth=2, label=f"MACRO (AP={ap_macro:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (Macro)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# ---------------- MAIN ----------------
def main():
    print("[STEP] Loading dataset + extracting YAMNet embeddings...")
    X, y = load_dataset(DATA_DIR)
    print("[INFO] X shape:", X.shape)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Outputs folder: {out_dir}")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("[INFO] Classes:", le.classes_)

    # Pipeline classifier
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            multi_class="auto",
            n_jobs=-1
        ))
    ])

    # CV scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("[STEP] 5-fold cross-validation (embeddings + LogReg)...")
    cv_scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")
    print("[CV] Accuracy per fold:", cv_scores)
    print("[CV] Mean accuracy:", float(cv_scores.mean()))

    cv_df = pd.DataFrame({"fold": np.arange(1, len(cv_scores) + 1), "accuracy": cv_scores})
    cv_df.to_csv(os.path.join(out_dir, "cv_scores.csv"), index=False)
    plt.figure(figsize=(6, 4))
    plt.plot(cv_df["fold"], cv_df["accuracy"], marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Accuracy per Fold")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "cv_scores.png"), bbox_inches="tight")
    plt.close()

    # Learning curve (classification)
    print("[STEP] Learning curve (classification)...")
    lc_csv = os.path.join(out_dir, "learning_curve_classification.csv")
    lc_png = os.path.join(out_dir, "learning_curve_classification.png")
    plot_learning_curve_classification(clf, X, y_enc, cv=cv, out_csv=lc_csv, out_png=lc_png)
    print(f"[SAVED] {lc_csv}")
    print(f"[SAVED] {lc_png}")

    # Requested regression plots (not applicable)
    save_na_placeholder(
        "Learning Curve (Regression) — N/A",
        os.path.join(out_dir, "learning_curve_regression.png"),
        "This project is a classification task (Confident/Hesitant/Nervous).\nRegression learning curve is not applicable."
    )
    save_na_placeholder(
        "Residuals (Regression) — N/A",
        os.path.join(out_dir, "residuals_regression.png"),
        "Residual plots are for regression outputs.\nFor classification, use confusion matrix / PR curve instead."
    )

    # Train/test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("[STEP] Training final model (embeddings + LogReg)...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Probabilities for PR curve (LogReg supports predict_proba)
    y_proba = clf.predict_proba(X_test)

    # Metrics table (accuracy table diagram)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    p_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    r_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

    metrics_df = pd.DataFrame({
        "value": [acc, bacc, f1_macro, f1_weighted, p_macro, r_macro]
    }, index=[
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro"
    ])

    metrics_csv = os.path.join(out_dir, "accuracy_table.csv")
    metrics_df.to_csv(metrics_csv)
    metrics_png = os.path.join(out_dir, "accuracy_table.png")
    save_table_image(metrics_df, "Accuracy / Summary Metrics", metrics_png, font_size=10)
    print(f"[SAVED] Accuracy table CSV -> {metrics_csv}")
    print(f"[SAVED] Accuracy table PNG -> {metrics_png}")

    # Classification report
    report_txt = classification_report(y_test, y_pred, target_names=le.classes_)
    print("\n[REPORT]\n", report_txt)

    report_dict = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T
    report_csv = os.path.join(out_dir, "classification_report.csv")
    report_df.to_csv(report_csv)
    report_png = os.path.join(out_dir, "classification_report.png")
    save_table_image(report_df, "Classification Report", report_png, font_size=8)
    print(f"[SAVED] Classification report CSV -> {report_csv}")
    print(f"[SAVED] Classification report PNG -> {report_png}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax_cm, colorbar=True)
    plt.title("Confusion Matrix")
    cm_png = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_png, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Confusion matrix PNG -> {cm_png}")

    # Precision-Recall macro
    pr_csv = os.path.join(out_dir, "precision_recall_macro.csv")
    pr_png = os.path.join(out_dir, "precision_recall_macro.png")
    plot_precision_recall_macro(y_test, y_proba, le.classes_, pr_csv, pr_png)
    print(f"[SAVED] PR macro CSV -> {pr_csv}")
    print(f"[SAVED] PR macro PNG -> {pr_png}")

    # Save y_true / y_pred
    y_df = pd.DataFrame({
        "y_true_idx": y_test,
        "y_true_label": le.inverse_transform(y_test),
        "y_pred_idx": y_pred,
        "y_pred_label": le.inverse_transform(y_pred),
    })
    y_csv = os.path.join(out_dir, "y_true_y_pred.csv")
    y_df.to_csv(y_csv, index=False)
    print(f"[SAVED] y_true / y_pred -> {y_csv}")

    # Save model bundle
    model_bundle = {
        "model": clf,  # scaler + logistic regression
        "label_encoder": le,
        "sample_rate": SAMPLE_RATE,
        "embedding_backend": "yamnet",
        "yamnet_handle": YAMNET_HANDLE,
        "embed_pool": EMBED_POOL,
        "feature_dim": int(X.shape[1]),
    }

    # Root save (for app.py)
    joblib.dump(model_bundle, ROOT_MODEL_PATH)
    print(f"\n[SAVED] Model (root) -> {ROOT_MODEL_PATH}")

    # Run folder save
    run_model_path = os.path.join(out_dir, "voice_confidence_model.joblib")
    joblib.dump(model_bundle, run_model_path)
    print(f"[SAVED] Model (run folder) -> {run_model_path}")


if __name__ == "__main__":
    main()
