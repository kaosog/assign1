# HW1: Feature Extraction from Face Images (AR 22-point .pts landmarks).
# Computes 7 required ratios per image and writes `outputs/features.csv`.
import csv
import math
import re
from pathlib import Path


_FLOAT_PAIR_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")

_FEATURE_NAMES = [
    "eye_length_ratio",
    "eye_distance_ratio",
    "nose_ratio",
    "lip_size_ratio",
    "lip_length_ratio",
    "eyebrow_length_ratio",
    "aggressive_ratio",
]

# Read 22 (x,y) landmarks from a .pts file.
def load_pts(path, expected_points=22):
    pts = []
    for line in path.read_text(errors="ignore").splitlines():
        m = _FLOAT_PAIR_RE.search(line)
        if m:
            pts.append((float(m.group(1)), float(m.group(2))))
    if len(pts) != expected_points:
        raise ValueError(f"{path.name}: expected {expected_points} points, found {len(pts)}")
    return pts

# Euclidean distance between two 2D points.
def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

# Distance between 0-indexed landmark points (0..21).
def d(points, i, j):

    return dist(points[i], points[j])


# Compute the 7 required feature ratios.
def extract_features(points):
    W = d(points, 8, 13)
    N = d(points, 20, 21)

    eye_len_left = d(points, 9, 10)
    eye_len_right = d(points, 11, 12)


    return {
        "eye_length_ratio": max(eye_len_left, eye_len_right) / W,
        "eye_distance_ratio": d(points, 0, 1) / W,
        "nose_ratio": d(points, 15, 16) / N,
        "lip_size_ratio": d(points, 2, 3) / d(points, 17, 18),
        "lip_length_ratio": d(points, 2, 3) / N,
        "eyebrow_length_ratio": max(d(points, 4, 5), d(points, 6, 7)) / W,
        "aggressive_ratio": d(points, 10, 19) / N,
    }

# Recursively find all .pts files under root.
# Dedupe because Windows paths/globs are case-insensitive.
def find_pts_files(root):
    seen = set()
    out = []
    for pattern in ("*.pts", "*.PTS"):
        for p in root.rglob(pattern):
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return sorted(out)

# Extract features for all images and write a CSV.
def main():
    script_dir = Path(__file__).resolve().parent

    root = Path("Face Database")
    if not root.is_absolute():
        root = (script_dir / root).resolve()

    out_csv = Path("outputs/features.csv")
    if not out_csv.is_absolute():
        out_csv = (script_dir / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for pts_path in find_pts_files(root):
        points = load_pts(pts_path, expected_points=22)
        feats = extract_features(points)
        rows.append(
            {
                "person_id": pts_path.parent.name,
                "image_id": pts_path.stem,
                **feats,
            }
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["person_id", "image_id", *_FEATURE_NAMES])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
