"""
01_regularization.py - Ridge vs Lasso (Roadmap Days 62-66)

Regularization is a way to stop a model from trusting any single input too much.
It adds a "penalty" for big weights (coefficients). Ridge and Lasso are two
flavors of that penalty.
"""

import matplotlib

matplotlib.use("Agg")  # save charts to files; no pop-up window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score

from data import get_regression_data, output_path


def main():
    # Get the regression data (200 rows, 30 columns, only 5 really useful).
    X_train, X_test, y_train, y_test = get_regression_data()

    # Three models: one with no penalty, and two with penalties.
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),  # alpha = how strong the penalty is
        "Lasso": Lasso(alpha=0.1),
    }

    print("Test R^2 scores (1.0 = perfect, 0 = no better than guessing the average):")
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)  # learn from the training rows
        preds = model.predict(X_test)  # predict the hidden test rows
        scores[name] = r2_score(y_test, preds)  # grade the predictions
        print(f"  {name:<7} R^2 = {scores[name]:.3f}")

    # Lasso can push weights to exactly zero. Count how many it zeroed out.
    lasso_coefs = models["Lasso"].coef_
    zero_count = int(np.sum(lasso_coefs == 0))
    print(f"\nLasso set {zero_count} of 30 coefficients to exactly zero.")

    print("\nPlain-English takeaway:")
    print("  Ridge shrinks every weight a little, but keeps them all.")
    print(
        "  Lasso can shrink some weights all the way to zero, which removes those inputs."
    )

    # Bar chart: size of each weight (absolute value) for each model.
    # Taking absolute value means we compare how big the weights are, ignoring sign.
    feature_positions = np.arange(30)
    width = 0.27  # width of each bar
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, name in enumerate(models):
        coefs = np.abs(models[name].coef_)
        ax.bar(feature_positions + (i - 1) * width, coefs, width, label=name)
    ax.set_xlabel("Input feature number")
    ax.set_ylabel("Size of weight (absolute value)")
    ax.set_title("Ridge shrinks weights; Lasso can zero them out")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path("regularization.png"), dpi=120)
    plt.close(fig)
    print("\nChart saved to outputs/regularization.png")

    return {
        "topic": "Ridge vs Lasso (Days 62-66)",
        "key_number": f"Lasso zeroed {zero_count}/30 weights",
        "takeaway": "Lasso removes useless inputs on its own; Ridge only shrinks them.",
    }


if __name__ == "__main__":
    main()
