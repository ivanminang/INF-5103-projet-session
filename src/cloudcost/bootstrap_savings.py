
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# CLOUDCOST ANALYZER V3 - BOOTSTRAP DES ECONOMIES
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "vm_best_recommendations_v3.csv"
OUTPUT_FILE = DATA_DIR / "bootstrap_results_v3.csv"
SAMPLES_FILE = DATA_DIR / "bootstrap_samples_v3.csv"

N_BOOTSTRAP = 5000
SEED = 42
ALPHA = 0.05


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE, dtype={"vm_id": str})

    required = [
        "vm_id",
        "current_cost",
        "optimized_cost",
        "monthly_savings",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    if df["vm_id"].duplicated().any():
        raise ValueError("Identifiants VM dupliques.")

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = (
        df[required[1:]].notna().all(axis=1)
        & np.isfinite(df[required[1:]]).all(axis=1)
        & (df["current_cost"] > 0)
        & (df["optimized_cost"] >= 0)
    )

    excluded = int((~valid).sum())
    df = df.loc[valid].copy()

    if df.empty:
        raise ValueError("Aucune VM avec couts valides.")

    # Recalculer les economies a partir des couts.
    df["monthly_savings"] = (
        df["current_cost"] - df["optimized_cost"]
    )

    current = df["current_cost"].to_numpy(float)
    optimized = df["optimized_cost"].to_numpy(float)
    savings = df["monthly_savings"].to_numpy(float)

    n = len(df)
    rng = np.random.default_rng(SEED)

    boot_total = np.empty(N_BOOTSTRAP)
    boot_pct = np.empty(N_BOOTSTRAP)
    boot_mean = np.empty(N_BOOTSTRAP)

    # Reechantillonnage par VM, pas par observation
    # temporelle, pour conserver l'unite statistique.
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)

        sampled_current = current[idx].sum()
        sampled_savings = savings[idx].sum()

        boot_total[i] = sampled_savings
        boot_mean[i] = sampled_savings / n

        boot_pct[i] = (
            100 * sampled_savings / sampled_current
        )

    total_current = current.sum()
    total_optimized = optimized.sum()
    total_savings = savings.sum()
    savings_pct = 100 * total_savings / total_current

    low = ALPHA / 2
    high = 1 - ALPHA / 2

    total_ci = np.quantile(boot_total, [low, high])
    pct_ci = np.quantile(boot_pct, [low, high])
    mean_ci = np.quantile(boot_mean, [low, high])

    results = pd.DataFrame([{
        "vm_count": n,
        "excluded_vm": excluded,
        "bootstrap_repetitions": N_BOOTSTRAP,
        "seed": SEED,
        "current_cost": total_current,
        "optimized_cost": total_optimized,
        "monthly_savings": total_savings,
        "monthly_savings_ci_low": total_ci[0],
        "monthly_savings_ci_high": total_ci[1],
        "savings_pct": savings_pct,
        "savings_pct_ci_low": pct_ci[0],
        "savings_pct_ci_high": pct_ci[1],
        "mean_savings_per_vm": savings.mean(),
        "mean_savings_ci_low": mean_ci[0],
        "mean_savings_ci_high": mean_ci[1],
    }])

    samples = pd.DataFrame({
        "replication": np.arange(1, N_BOOTSTRAP + 1),
        "monthly_savings": boot_total,
        "savings_pct": boot_pct,
        "mean_savings_per_vm": boot_mean,
    })

    results.to_csv(OUTPUT_FILE, index=False)
    samples.to_csv(SAMPLES_FILE, index=False)

    print("\n===== BOOTSTRAP CLOUDCOST V3 =====")
    print("VM analysees :", n)
    print("VM exclues :", excluded)
    print("Replications :", N_BOOTSTRAP)

    print("\nEconomies mensuelles simulees :")
    print(f"Estimation : {total_savings:,.2f} USD")
    print(
        "IC bootstrap 95 % : "
        f"[{total_ci[0]:,.2f} ; {total_ci[1]:,.2f}] USD"
    )

    print("\nPourcentage d'economies :")
    print(f"Estimation : {savings_pct:.2f} %")
    print(
        "IC bootstrap 95 % : "
        f"[{pct_ci[0]:.2f} ; {pct_ci[1]:.2f}] %"
    )

    print("\nEconomie moyenne par VM :")
    print(f"Estimation : {savings.mean():.2f} USD")
    print(
        "IC bootstrap 95 % : "
        f"[{mean_ci[0]:.2f} ; {mean_ci[1]:.2f}] USD"
    )

    print("\nFichiers crees :")
    print(OUTPUT_FILE)
    print(SAMPLES_FILE)


if __name__ == "__main__":
    main()
