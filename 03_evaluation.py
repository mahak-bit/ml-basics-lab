"""
03_evaluation.py - Confusion matrix, precision, recall, F1 (Days 75-76)

Accuracy alone can lie when one class is rare. This script shows why we need
more ways to grade a classifier.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data import get_classification_data


def main():
    X_train, X_test, y_train, y_test = get_classification_data()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy: the share of rows we got right, overall.
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

    # Confusion matrix: a 2x2 table of right and wrong answers.
    # ravel() unpacks it into four numbers in this order: TN, FP, FN, TP.
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("\nConfusion matrix (rows = real answer, columns = predicted):")
    print(f"  TN (real 0, predicted 0, correct): {tn}")
    print(f"  FP (real 0, predicted 1, false alarm): {fp}")
    print(f"  FN (real 1, predicted 0, missed it):   {fn}")
    print(f"  TP (real 1, predicted 1, caught it):   {tp}")

    # Precision: of the rows we called 1, how many really were 1?
    # Recall: of the real 1s, how many did we catch?
    # F1: one number that balances precision and recall.
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    # --- The trap: a lazy model that always guesses the majority class ---
    dumb = DummyClassifier(strategy="most_frequent")
    dumb.fit(X_train, y_train)
    dumb_pred = dumb.predict(X_test)
    dumb_acc = accuracy_score(y_test, dumb_pred)
    dumb_recall = recall_score(y_test, dumb_pred, zero_division=0)
    print("\n--- The trap ---")
    print("A dumb model that always answers 'class 0' (the majority):")
    print(f"  accuracy = {dumb_acc:.3f}")
    print(f"  recall   = {dumb_recall:.3f}")
    print("High accuracy can still mean ZERO recall: it never finds a single real 1.")

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    return {
        "topic": "Evaluation metrics (Days 75-76)",
        "key_number": f"Dummy accuracy {dumb_acc:.2f} but recall {dumb_recall:.2f}",
        "takeaway": "A high accuracy score can hide a model that misses every rare case.",
    }


if __name__ == "__main__":
    main()
