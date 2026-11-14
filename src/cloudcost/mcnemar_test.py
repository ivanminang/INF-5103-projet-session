
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

# =====================================================
# CLOUDCOST ANALYZER V3 - TEST DE McNEMAR
# Comparaison des méthodes A et B sur les mêmes VM
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "vm_backtest_validation_v3.csv"
OUTPUT_FILE = DATA_DIR / "mcnemar_results_v3.csv"
PAIRS_FILE = DATA_DIR / "mcnemar_pairs_v3.csv"

RISK_THRESHOLD = 0.01
MIN_VALID_OBSERVATIONS = 100
ALPHA = 0.05


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE, dtype={"vm_id": str})

    required = [
        "vm_id",
        "method",
        "global_risk",
        "valid_observations",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df = df[df["method"].isin(["A", "B"])].copy()

    # Ne comparer que les VM disposant de suffisamment
    # d'observations valides pour les deux méthodes.
    df["eligible"] = (
        df["global_risk"].notna()
        & (df["valid_observations"] >= MIN_VALID_OBSERVATIONS)
    )

    df["admissible"] = (
        df["eligible"]
        & (df["global_risk"] <= RISK_THRESHOLD)
    )

    if df.duplicated(["vm_id", "method"]).any():
        raise ValueError(
            "Plusieurs résultats pour une même VM et méthode."
        )

    wide = df.pivot(
        index="vm_id",
        columns="method",
        values=["eligible", "admissible", "global_risk"]
    )

    # Les deux méthodes doivent être présentes.
    wide = wide.dropna(
        subset=[
            ("eligible", "A"),
            ("eligible", "B"),
        ]
    )

    # Exclure les VM insuffisamment documentées.
    wide = wide[
        wide[("eligible", "A")].astype(bool)
        & wide[("eligible", "B")].astype(bool)
    ].copy()

    a = wide[("admissible", "A")].astype(bool)
    b = wide[("admissible", "B")].astype(bool)

    # Tableau de contingence apparié.
    both_yes = int((a & b).sum())
    a_yes_b_no = int((a & ~b).sum())
    a_no_b_yes = int((~a & b).sum())
    both_no = int((~a & ~b).sum())

    discordant = a_yes_b_no + a_no_b_yes

    # Test exact bilatéral de McNemar.
    if discordant > 0:
        p_value = binomtest(
            min(a_yes_b_no, a_no_b_yes),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    else:
        p_value = 1.0

    a_rate = float(a.mean() * 100)
    b_rate = float(b.mean() * 100)

    result = pd.DataFrame([{
        "split": "validation",
        "vm_compared": len(wide),
        "both_admissible": both_yes,
        "a_only_admissible": a_yes_b_no,
        "b_only_admissible": a_no_b_yes,
        "neither_admissible": both_no,
        "discordant_pairs": discordant,
        "a_admissible_pct": a_rate,
        "b_admissible_pct": b_rate,
        "difference_b_minus_a_pp": b_rate - a_rate,
        "p_value": p_value,
        "alpha": ALPHA,
        "significant": p_value < ALPHA,
    }])

    pairs = pd.DataFrame({
        "vm_id": wide.index,
        "a_admissible": a.to_numpy(),
        "b_admissible": b.to_numpy(),
        "a_risk_pct": (
            wide[("global_risk", "A")].to_numpy() * 100
        ),
        "b_risk_pct": (
            wide[("global_risk", "B")].to_numpy() * 100
        ),
    })

    result.to_csv(OUTPUT_FILE, index=False)
    pairs.to_csv(PAIRS_FILE, index=False)

    print("\n===== TEST EXACT DE McNEMAR =====")
    print("VM comparees :", len(wide))

    print("\nTableau de contingence :")
    print("A oui / B oui :", both_yes)
    print("A oui / B non :", a_yes_b_no)
    print("A non / B oui :", a_no_b_yes)
    print("A non / B non :", both_no)

    print("\nTaux d'admissibilite :")
    print("Methode A :", round(a_rate, 2), "%")
    print("Methode B :", round(b_rate, 2), "%")

    print("\nP-value :", f"{p_value:.6g}")

    if p_value < ALPHA:
        print(
            "Conclusion : difference statistiquement "
            "significative au seuil de 5 %."
        )
    else:
        print(
            "Conclusion : difference non statistiquement "
            "significative au seuil de 5 %."
        )

    print("\nFichiers crees :")
    print(OUTPUT_FILE)
    print(PAIRS_FILE)


if __name__ == "__main__":
    main()
