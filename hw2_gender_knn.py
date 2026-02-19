"""HW2: Gender classification from extracted face features using KNN.

Loads `outputs/features.csv` produced by HW1, performs a person-level split
(3 male + 3 female persons for train; remaining persons for test), trains
KNN, and reports confusion matrix, accuracy, precision, and recall.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "eye_length_ratio",
    "eye_distance_ratio",
    "nose_ratio",
    "lip_size_ratio",
    "lip_length_ratio",
    "eyebrow_length_ratio",
    "aggressive_ratio",
]


def gender_from_person_id(person_id: str) -> str:
    pid = str(person_id).strip().lower()
    if pid.startswith("m-"):
        return "male"
    if pid.startswith("w-"):
        return "female"
    raise ValueError(f"Cannot infer gender from person_id='{person_id}'")


def build_person_split(df: pd.DataFrame, male_train_count: int = 3, female_train_count: int = 3):
    male_ids = sorted(df.loc[df["gender"] == "male", "person_id"].unique().tolist())
    female_ids = sorted(df.loc[df["gender"] == "female", "person_id"].unique().tolist())

    if len(male_ids) < male_train_count or len(female_ids) < female_train_count:
        raise ValueError(
            "Not enough people per class for requested train split: "
            f"males={len(male_ids)}, females={len(female_ids)}"
        )

    train_people = set(male_ids[:male_train_count] + female_ids[:female_train_count])
    test_people = set(male_ids[male_train_count:] + female_ids[female_train_count:])

    train_df = df[df["person_id"].isin(train_people)].copy()
    test_df = df[df["person_id"].isin(test_people)].copy()
    return train_df, test_df, sorted(train_people), sorted(test_people)


def evaluate_k(x_train, y_train, x_test, y_test, k: int):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return {
        "k": k,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="male", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label="male", zero_division=0),
        "y_pred": y_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HW2 KNN gender classification from feature CSV")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("outputs/features.csv"),
        help="Path to HW1 feature CSV (default: outputs/features.csv relative to this script)",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of neighbors for KNN (default: 3)")
    parser.add_argument(
        "--k-values",
        type=str,
        default="",
        help="Comma-separated k values to tune (example: 1,3,5,7). "
        "If provided, best k is selected by accuracy, then precision, then recall.",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("outputs/hw2_test_predictions.csv"),
        help="Where to write test predictions CSV",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    features_csv = args.features_csv
    if not features_csv.is_absolute():
        features_csv = (script_dir / features_csv).resolve()
    if not features_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {features_csv}")

    pred_out = args.predictions_out
    if not pred_out.is_absolute():
        pred_out = (script_dir / pred_out).resolve()
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    required = {"person_id", "image_id", *FEATURE_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {features_csv.name}: {sorted(missing)}")

    df["gender"] = df["person_id"].map(gender_from_person_id)

    train_df, test_df, train_people, test_people = build_person_split(df)

    x_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["gender"].to_numpy()
    x_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df["gender"].to_numpy()

    max_k = len(train_df)
    if args.k_values.strip():
        requested = [int(v.strip()) for v in args.k_values.split(",") if v.strip()]
    else:
        requested = [args.k]

    valid_ks = sorted(set(k for k in requested if k >= 1 and k <= max_k))
    if not valid_ks:
        raise ValueError(f"No valid k values. Must be between 1 and {max_k}. Got: {requested}")

    results = [evaluate_k(x_train, y_train, x_test, y_test, k) for k in valid_ks]
    best = max(results, key=lambda r: (r["accuracy"], r["precision"], r["recall"], -r["k"]))
    chosen_k = best["k"]
    y_pred = best["y_pred"]

    labels = ["male", "female"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    acc = best["accuracy"]
    precision = best["precision"]
    recall = best["recall"]

    print("=== HW2 Gender Classification (KNN) ===")
    print(f"features_csv : {features_csv}")
    if len(valid_ks) > 1:
        print(f"k_candidates : {valid_ks}")
    print(f"k_selected   : {chosen_k}")
    print(f"train_people : {train_people}")
    print(f"test_people  : {test_people}")
    print(f"train_rows   : {len(train_df)}")
    print(f"test_rows    : {len(test_df)}")
    if len(valid_ks) > 1:
        print()
        print("K tuning results:")
        print("k\taccuracy\tprecision\trecall")
        for r in results:
            print(f"{r['k']}\t{r['accuracy']:.4f}\t{r['precision']:.4f}\t{r['recall']:.4f}")
    print()
    print("Confusion Matrix (rows=true, cols=pred, labels=[male, female]):")
    print(cm)
    print()
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f} (male as positive class)")
    print(f"Recall    : {recall:.4f} (male as positive class)")

    out_df = test_df[["person_id", "image_id", "gender"]].rename(columns={"gender": "actual_gender"})
    out_df["predicted_gender"] = y_pred
    out_df.to_csv(pred_out, index=False)
    print(f"\nWrote predictions: {pred_out}")


if __name__ == "__main__":
    main()
