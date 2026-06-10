import pandas as pd

from .config import EXPECTED_GENES, EXPECTED_MODELS, EXPECTED_REPS, EXPECTED_ROWS


def print_dataset_checks(df: pd.DataFrame):
    print("\nDATASET CHECKS")
    print("Rows:", len(df))
    print("Expected rows:", EXPECTED_ROWS)
    print("Unique reps:", df["rep"].nunique())
    print("Expected reps:", len(EXPECTED_REPS))

    print("\nRows per rep:")
    rows_per_rep = df.groupby("rep").size()
    print(rows_per_rep.to_string())

    bad_reps = rows_per_rep[rows_per_rep != EXPECTED_MODELS * EXPECTED_GENES]

    print("\nBAD REPS:")
    if bad_reps.empty:
        print("None")
    else:
        print(bad_reps.to_string())

    print("\nRows per model:")
    print(df.groupby("model").size().to_string())

    print("\nMissing reps by model/gene:")
    actual = df.groupby(["model", "gene"])["rep"].apply(set).reset_index()

    actual["missing_reps"] = actual["rep"].apply(
        lambda reps: sorted(EXPECTED_REPS - reps)
    )

    actual["num_missing"] = actual["missing_reps"].apply(len)

    missing = actual[actual["num_missing"] > 0][
        ["model", "gene", "num_missing", "missing_reps"]
    ]

    if missing.empty:
        print("None")
    else:
        print(missing.to_string(index=False))

