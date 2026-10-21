"""
run_all.py - runs every part of the lab in order, then prints a summary table.

Usage:  python run_all.py
"""

import importlib
import os

import pandas as pd

from data import OUTPUT_DIR

# File names start with numbers, so we load them with importlib instead of "import".
PARTS = [
    "01_regularization",
    "02_classification",
    "03_evaluation",
    "04_decision_tree",
    "05_kmeans",
]


def main():
    # Make sure the outputs/ folder exists before any script saves a chart.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for name in PARTS:
        print("=" * 70)
        print(f"Running {name}")
        print("=" * 70)
        module = importlib.import_module(name)
        results.append(module.main())  # each part returns a short summary dict
        print()

    # Summary table: one row per part.
    summary = pd.DataFrame(results, columns=["topic", "key_number", "takeaway"])
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for _, row in summary.iterrows():
        print(f"Topic:     {row['topic']}")
        print(f"Key number: {row['key_number']}")
        print(f"Takeaway:  {row['takeaway']}")
        print()


if __name__ == "__main__":
    main()
