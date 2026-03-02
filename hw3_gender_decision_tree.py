"""HW3: Gender classification from extracted face features using Decision Tree.

Loads `outputs/features.csv` produced by HW1, performs the same person-level
split used in HW2 (3 male + 3 female persons for train; remaining persons for
test), trains a Decision Tree classifier, and reports confusion matrix,
accuracy, precision, and recall.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

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


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    features_csv = Path("outputs/features.csv")
    hw3_out_dir = Path("outputs/hw3")

    if not features_csv.is_absolute():
        features_csv = (script_dir / features_csv).resolve()
    if not features_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {features_csv}")

    if not hw3_out_dir.is_absolute():
        hw3_out_dir = (script_dir / hw3_out_dir).resolve()
    hw3_out_dir.mkdir(parents=True, exist_ok=True)

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

    dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)

    labels = ["male", "female"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="male", zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label="male", zero_division=0)

    print("HW3 Decision Tree results")
    print("Confusion Matrix (rows=true, cols=pred, labels=[male, female]):")
    print(cm)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f} (male as positive class)")
    print(f"Recall    : {recall:.4f} (male as positive class)")

    pred_out = hw3_out_dir / "hw3_test_predictions_decision_tree.csv"
    pred_df = test_df[["person_id", "image_id", "gender"]].rename(columns={"gender": "actual_gender"})
    pred_df["predicted_gender"] = y_pred
    pred_df.to_csv(pred_out, index=False)

    metrics_out = hw3_out_dir / "hw3_decision_tree_metrics.csv"
    metrics_df = pd.DataFrame(
        [
            {
                "criterion": "entropy",
                "random_state": 42,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "tree_depth": dt.get_depth(),
                "leaf_nodes": dt.get_n_leaves(),
                "accuracy": round(acc, 4),
                "precision_male": round(precision, 4),
                "recall_male": round(recall, 4),
                "cm_true_male_pred_male": int(cm[0, 0]),
                "cm_true_male_pred_female": int(cm[0, 1]),
                "cm_true_female_pred_male": int(cm[1, 0]),
                "cm_true_female_pred_female": int(cm[1, 1]),
            }
        ]
    )
    metrics_df.to_csv(metrics_out, index=False)

    fi_out = hw3_out_dir / "hw3_feature_importance.csv"
    fi_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": dt.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    fi_df.to_csv(fi_out, index=False)

    print("\nSaved:")
    print(f"- {metrics_out}")
    print(f"- {pred_out}")
    print(f"- {fi_out}")


if __name__ == "__main__":
    main()
