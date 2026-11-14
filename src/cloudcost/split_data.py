
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "bitbrains_normalized.parquet"

RATIOS = {
    "train": 0.60,
    "validation": 0.20,
    "test": 0.20,
}


def main():
    df = pd.read_parquet(INPUT_FILE)

    df["vm_id"] = df["vm_id"].astype(str)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )

    # Tri chronologique à l'intérieur de chaque VM
    df = df.sort_values(
        ["vm_id", "timestamp"]
    ).reset_index(drop=True)

    # Rang chronologique et taille de chaque VM
    groups = df.groupby("vm_id", sort=False)
    rank = groups.cumcount()
    count = groups["vm_id"].transform("size")

    train_end = (count * RATIOS["train"]).astype(int)
    val_end = (
        count * (
            RATIOS["train"] + RATIOS["validation"]
        )
    ).astype(int)

    train = df[rank < train_end].copy()

    validation = df[
        (rank >= train_end) & (rank < val_end)
    ].copy()

    test = df[rank >= val_end].copy()

    for name, subset in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        path = DATA_DIR / f"bitbrains_{name}_v3.parquet"
        subset.to_parquet(path, index=False)

        print(
            f"{name}: {len(subset):,} observations, "
            f"{subset['vm_id'].nunique()} VM"
        )

    print("\nTotal initial :", len(df))
    print(
        "Total réparti :",
        len(train) + len(validation) + len(test)
    )


if __name__ == "__main__":
    main()
