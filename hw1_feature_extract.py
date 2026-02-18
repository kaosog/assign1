#!/usr/bin/env python3
"""
HW1: Feature Extraction from Face Images
CSCI 405 / CIS 605

Author: <YOUR NAME>
Date: <DATE>

This script extracts the 7 required features from AR 22-point .pts files and
writes a CSV you can paste into your report.

How to run (NO required args):
- Put this file in the same folder as "Face Database/"
- Run:
    python3 hw1_feature_extract.py

Optional:
    python3 hw1_feature_extract.py --root "/full/path/to/Face Database" --out_dir outputs

Feature definitions (1-indexed points):
1) Eye length ratio      = max( d(9,10), d(11,12) ) / d(8,13)
2) Eye distance ratio    = d( center_left_eye, center_right_eye ) / d(8,13)
   - centers are midpoints of (9,10) and (11,12)
3) Nose ratio            = d(15,16) / d(20,21)
4) Lip size ratio        = d(2,3) / d(17,18)
5) Lip length ratio      = d(2,3) / d(20,21)
6) Eyebrow length ratio  = max(d(4,5), d(6,7)) / d(8,13)
7) Aggressive ratio      = d(10,19) / d(20,21)

Data format:
Each .pts file contains 22 (x,y) points. We extract every numeric "x y" pair in order.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Loading .pts files
# -----------------------------

_FLOAT_PAIR_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def load_pts(path: Path, expected_points: int = 22) -> np.ndarray:
    """Load a .pts file and return an array of shape (22, 2)."""
    pts: List[Tuple[float, float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        m = _FLOAT_PAIR_RE.search(line.strip())
        if m:
            pts.append((float(m.group(1)), float(m.group(2))))

    if len(pts) != expected_points:
        raise ValueError(f"{path.name}: expected {expected_points} points, found {len(pts)}")

    return np.array(pts, dtype=float)


# -----------------------------
# Geometry helpers
# -----------------------------

def euclid(p: np.ndarray, q: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(p - q))


def d(points: np.ndarray, i: int, j: int) -> float:
    """Distance between 1-indexed landmark points i and j."""
    return euclid(points[i - 1], points[j - 1])


def midpoint(points: np.ndarray, i: int, j: int) -> np.ndarray:
    """Midpoint between 1-indexed points i and j."""
    return (points[i - 1] + points[j - 1]) / 2.0


# -----------------------------
# Feature extraction
# -----------------------------

def extract_features(points: np.ndarray) -> Dict[str, float]:
    """Extract the 7 required features from the 22-point landmark array."""
    # Normalizers per assignment
    W = d(points, 8, 13)
    N = d(points, 20, 21)

    # Eyes (from Figure 1 numbering)
    eye_len_left = d(points, 9, 10)
    eye_len_right = d(points, 11, 12)
    c_left = midpoint(points, 9, 10)
    c_right = midpoint(points, 11, 12)

    return {
        "eye_length_ratio": max(eye_len_left, eye_len_right) / W,
        "eye_distance_ratio": euclid(c_left, c_right) / W,
        "nose_ratio": d(points, 15, 16) / N,
        "lip_size_ratio": d(points, 2, 3) / d(points, 17, 18),
        "lip_length_ratio": d(points, 2, 3) / N,
        "eyebrow_length_ratio": max(d(points, 4, 5), d(points, 6, 7)) / W,
        "aggressive_ratio": d(points, 10, 19) / N,
    }


def find_pts_files(root: Path) -> List[Path]:
    """Find .pts files recursively (case-insensitive)."""
    return sorted(list(root.rglob("*.pts")) + list(root.rglob("*.PTS")))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HW1 feature extraction (no required args)")
    parser.add_argument(
        "--root",
        default="Face Database",
        help='Dataset root folder (default: "Face Database" next to this script)',
    )
    parser.add_argument(
        "--out_dir",
        default="outputs",
        help='Output folder (default: "outputs")',
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    # If root is relative, interpret it relative to this script's folder.
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise SystemExit(
            f'Root folder not found: {root}\n'
            f'Put "Face Database" next to this script, or run:\n'
            f'  python3 {Path(__file__).name} --root "/full/path/to/Face Database"'
        )

    pts_files = find_pts_files(root)
    if not pts_files:
        raise SystemExit(f"No .pts files found under: {root}")

    rows: List[Dict] = []
    errors: List[str] = []

    for pts_path in pts_files:
        try:
            points = load_pts(pts_path, expected_points=22)
            feats = extract_features(points)

            person_id = pts_path.parent.name  # e.g., "m-001"
            image_id = pts_path.stem          # e.g., "m-001-01"

            rows.append({"person_id": person_id, "image_id": image_id, **feats})
        except Exception as e:
            errors.append(f"{pts_path}: {e}")

    df = pd.DataFrame(rows).sort_values(["person_id", "image_id"])

    out_csv = out_dir / "features.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote: {out_csv}  ({len(df)} images)")

    if errors:
        err_path = out_dir / "errors.txt"
        err_path.write_text("\n".join(errors))
        print(f"⚠️ {len(errors)} file(s) had errors. See: {err_path}")


if __name__ == "__main__":
    main()
