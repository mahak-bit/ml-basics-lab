"""
02_classification.py - Logistic regression, sigmoid, and cross-entropy (Days 69-74)

Classification = predicting a category (yes/no) instead of a number.
Logistic regression gives a probability between 0 and 1 for "yes".
The sigmoid function is what squashes any number into that 0-to-1 range.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from data import get_classification_data


def sigmoid(z):
    """
    Squash any number into a value between 0 and 1.
    Large positive z -> close to 1. Large negative z -> close to 0. z = 0 -> 0.5.
    """
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true, y_prob):
    """
    Measure how wrong the probabilities are. Smaller is better.

    For each row:
      - If the true answer is 1, we are rewarded for a probability near 1.
      - If the true answer is 0, we are rewarded for a probability near 0.
    A confident WRONG answer gets a huge penalty, because -log(tiny number) is huge.
    """
    y_true = np.asarray(y_true, dtype=float)
    # Keep probabilities away from exactly 0 or 1, so log() never breaks.
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)
    loss_per_row = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return loss_per_row.mean()  # average over all rows


def main():
    X_train, X_test, y_train, y_test = get_classification_data()

    # max_iter=1000 gives the solver enough steps to finish cleanly.
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Probability of class 1 for each test row (second column = "yes").
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("First 5 test rows:")
    print(f"  predicted probability of class 1: {np.round(probs[:5], 3)}")
    print(f"  predicted class:                  {preds[:5]}")
    print(f"  true class:                       {y_test[:5]}")

    # --- Check our sigmoid against the model's own numbers ---
    # decision_function gives the raw score z before squashing.
    z = model.decision_function(X_test[:5])
    print("\nSigmoid check:")
    print(f"  sigmoid(0) = {sigmoid(0):.2f}  (must be 0.5)")
    print(
        f"  our sigmoid(raw scores) vs sklearn: max difference = "
        f"{np.max(np.abs(sigmoid(z) - probs[:5])):.2e}  (should be ~0)"
    )

    # --- Cross-entropy on three hand-picked examples ---
    print("\nBinary cross-entropy on three examples (smaller = better):")
    examples = [(1, 0.9), (1, 0.1), (0, 0.95)]
    for y, p in examples:
        print(
            f"  true y={y}, predicted p={p:.2f} -> loss = {binary_cross_entropy([y], [p]):.3f}"
        )
    print("  The last two are WRONG answers. The 0.1 guess for a true 1 is punished,")
    print(
        "  and the (y=0, p=0.95) row is a confident mistake, so its loss is the biggest."
    )

    # --- Compare our loss with sklearn's loss on the whole test set ---
    our_loss = binary_cross_entropy(y_test, probs)
    sk_loss = log_loss(y_test, probs)
    print("\nWhole test set:")
    print(f"  our binary_cross_entropy: {our_loss:.5f}")
    print(f"  sklearn log_loss:         {sk_loss:.5f}  (should match closely)")

    return {
        "topic": "Logistic regression + sigmoid + BCE (Days 69-74)",
        "key_number": f"Our loss {our_loss:.4f} vs sklearn {sk_loss:.4f}",
        "takeaway": "Sigmoid turns scores into probabilities; BCE punishes confident wrong guesses hard.",
    }


if __name__ == "__main__":
    main()
