"""
04_decision_tree.py - Entropy vs Gini, and max_depth (Day 80)

A decision tree asks a chain of yes/no questions ("is feature 3 > 0.5?").
Two ways to pick the best question are "gini" and "entropy". Both measure
how mixed up a group is: 0 means pure (all one class), bigger means messier.
"""

import matplotlib

matplotlib.use("Agg")  # save charts to files; no pop-up window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from data import get_classification_data, output_path


def entropy(p):
    """
    Mixed-up-ness of a group with two classes.
    p = share of class 1 in the group. Returns 0 for a pure group, 1 for a 50/50 mix.
    """
    q = 1 - p  # share of class 0
    total = 0.0
    for share in (p, q):
        if share > 0:  # skip 0 * log(0), which we treat as 0
            total -= share * np.log2(share)
    return total


def gini(p):
    """
    Another mixed-up-ness score for two classes.
    Returns 0 for a pure group, 0.5 for a 50/50 mix.
    """
    q = 1 - p
    return 1 - (p**2 + q**2)


def main():
    X_train, X_test, y_train, y_test = get_classification_data()

    # --- Same depth limit (3 questions deep), two different ways to split ---
    print("Trees limited to max_depth=3:")
    shallow_gini = DecisionTreeClassifier(
        criterion="gini", max_depth=3, random_state=42
    )
    shallow_entropy = DecisionTreeClassifier(
        criterion="entropy", max_depth=3, random_state=42
    )
    shallow_gini.fit(X_train, y_train)
    shallow_entropy.fit(X_train, y_train)
    acc_gini = shallow_gini.score(X_test, y_test)
    acc_entropy = shallow_entropy.score(X_test, y_test)
    print(f"  gini    test accuracy: {acc_gini:.3f}")
    print(f"  entropy test accuracy: {acc_entropy:.3f}")

    # --- No depth limit: the tree keeps asking questions until each leaf is pure ---
    deep = DecisionTreeClassifier(random_state=42)  # no max_depth on purpose
    deep.fit(X_train, y_train)
    train_acc = deep.score(X_train, y_train)
    test_acc = deep.score(X_test, y_test)
    print("\nTree with NO depth limit:")
    print(f"  training accuracy: {train_acc:.3f}")
    print(f"  test accuracy:     {test_acc:.3f}")
    print(f"  gap = {train_acc - test_acc:.3f}")
    print("  A big gap means OVERFITTING: the tree memorized the training rows")
    print("  instead of learning a pattern that also works on new rows.")

    # --- Entropy and Gini by hand, for a 2-class group ---
    print("\nEntropy and Gini from scratch:")
    for p in (0.5, 1.0):
        print(f"  p={p}: entropy = {entropy(p):.3f}, gini = {gini(p):.3f}")
    # Sanity checks: these must hold.
    assert np.isclose(entropy(0.5), 1.0), "entropy(0.5) should be 1"
    assert np.isclose(gini(0.5), 0.5), "gini(0.5) should be 0.5"
    print("  Checks passed: entropy(0.5) = 1 and gini(0.5) = 0.5")
    print("  p=1.0 means every row is class 1 (a pure group), so both scores are 0.")

    # --- Draw the top of the tree (only 2 levels, so it is readable) ---
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_tree(
        shallow_gini,
        max_depth=2,
        feature_names=feature_names,
        class_names=["class 0", "class 1"],
        filled=True,  # color boxes by class
        ax=ax,
    )
    ax.set_title("Top of a decision tree (gini)")
    fig.savefig(output_path("decision_tree.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\nTree plot saved to outputs/decision_tree.png")

    return {
        "topic": "Decision trees: gini, entropy, max_depth (Day 80)",
        "key_number": f"Depth-limited test acc {acc_gini:.2f}; unlimited gap {train_acc - test_acc:.2f}",
        "takeaway": "An unlimited tree memorizes training data; limiting depth fixes the overfitting.",
    }


if __name__ == "__main__":
    main()
