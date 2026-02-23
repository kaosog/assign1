"""HW2: Gender classification from extracted face features using KNN.

Loads `outputs/features.csv` produced by HW1, performs a person-level split
(3 male + 3 female persons for train; remaining persons for test), trains
KNN, and reports confusion matrix, accuracy, precision, and recall.
"""

from __future__ import annotations

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
    script_dir = Path(__file__).resolve().parent
    features_csv = Path("outputs/features.csv")
    hw2_out_dir = Path("outputs/hw2")
    k_values = [1, 3, 5, 7, 9]

    if not features_csv.is_absolute():
        features_csv = (script_dir / features_csv).resolve()
    if not features_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {features_csv}")

    if not hw2_out_dir.is_absolute():
        hw2_out_dir = (script_dir / hw2_out_dir).resolve()
    hw2_out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    required = {"person_id", "image_id", *FEATURE_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {features_csv.name}: {sorted(missing)}")

    df["gender"] = df["person_id"].map(gender_from_person_id)

    train_df, test_df, _, _ = build_person_split(df)

    x_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["gender"].to_numpy()
    x_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df["gender"].to_numpy()

    max_k = len(train_df)
    valid_ks = sorted(set(k for k in k_values if k >= 1 and k <= max_k))
    if not valid_ks:
        raise ValueError(f"No valid k values. Must be between 1 and {max_k}. Got: {k_values}")

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
    print(f"k_selected   : {chosen_k}")
    print(f"k_tested     : {valid_ks}")
    print(f"train_rows   : {len(train_df)}")
    print(f"test_rows    : {len(test_df)}")
    print()
    print("Confusion Matrix (rows=true, cols=pred, labels=[male, female]):")
    print(cm)
    print()
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f} (male as positive class)")
    print(f"Recall    : {recall:.4f} (male as positive class)")

    base_pred_df = test_df[["person_id", "image_id", "gender"]].rename(columns={"gender": "actual_gender"})

    for r in results:
        pred_path = hw2_out_dir / f"hw2_test_predictions_k{r['k']}.csv"
        pred_df = base_pred_df.copy()
        pred_df["predicted_gender"] = r["y_pred"]
        pred_df["k_used"] = r["k"]
        pred_df.to_csv(pred_path, index=False)

    best_pred_out = hw2_out_dir / f"hw2_test_predictions_best_k{chosen_k}.csv"
    best_out_df = base_pred_df.copy()
    best_out_df["predicted_gender"] = y_pred
    best_out_df["k_used"] = chosen_k
    best_out_df.to_csv(best_pred_out, index=False)

    results_df = pd.DataFrame(
        [
            {
                "k": r["k"],
                "accuracy": round(r["accuracy"], 4),
                "precision_male": round(r["precision"], 4),
                "recall_male": round(r["recall"], 4),
            }
            for r in results
        ]
    )
    results_out = hw2_out_dir / "hw2_k_results.csv"
    results_df.to_csv(results_out, index=False)

    print(f"\nWrote k results: {results_out}")
    print(f"Wrote best predictions: {best_pred_out}")
    print(f"Wrote per-k predictions in: {hw2_out_dir}")


if __name__ == "__main__":
    main()
