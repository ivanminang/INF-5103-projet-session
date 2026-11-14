
from pathlib import Path
import pandas as pd
import numpy as np

# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "bitbrains_normalized.parquet"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "vm_statistics.csv"
)


def calculate_metrics(values):
    """Calcule les statistiques d'une ressource."""

    x = pd.to_numeric(
        values, errors="coerce"
    ).dropna()

    if x.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "cv": np.nan,
            "min": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    mean = x.mean()
    std = x.std()

    return {
        "count": len(x),
        "mean": mean,
        "median": x.median(),
        "std": std,
        "cv": std / mean if mean != 0 else np.nan,
        "min": x.min(),
        "p90": x.quantile(0.90),
        "p95": x.quantile(0.95),
        "p99": x.quantile(0.99),
        "max": x.max(),
    }


def calculate_vm_statistics(df):
    """Calcule les statistiques CPU/RAM par VM."""

    results = []

    for vm_id, group in df.groupby("vm_id"):

        row = {
            "vm_id": vm_id,
            "observations": len(group),
            "cpu_cores": group["cpu_cores"].median(),
            "mem_alloc_gb": group["mem_alloc_gb"].median(),
        }

        for resource in ["cpu_pct", "mem_pct"]:

            metrics = calculate_metrics(
                group[resource]
            )

            for metric, value in metrics.items():
                row[f"{resource}_{metric}"] = value

        results.append(row)

    return pd.DataFrame(results)


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    stats = calculate_vm_statistics(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    stats.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===== STATISTIQUES BITBRAINS =====")
    print("VM analysées :", len(stats))

    print("\nAperçu des résultats :")

    columns = [
        "vm_id",
        "cpu_pct_mean",
        "cpu_pct_p95",
        "cpu_pct_p99",
        "mem_pct_mean",
        "mem_pct_p95",
        "mem_pct_p99",
    ]

    print(
        stats[columns]
        .head(10)
        .round(2)
        .to_string(index=False)
    )

    print("\nFichier créé :", OUTPUT_FILE)


if __name__ == "__main__":
    main()
