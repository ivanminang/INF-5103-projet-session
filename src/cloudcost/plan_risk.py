
from pathlib import Path
import numpy as np
import pandas as pd

# ==========================================================
# CLOUDCOST ANALYZER V3
# EVALUATION DU RISQUE DES FORFAITS COMMERCIAUX
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

BEST_FILE = DATA_DIR / "vm_best_recommendations_v3.csv"
OUTPUT_FILE = DATA_DIR / "vm_plan_risk_v3.csv"

RISK_THRESHOLD = 0.01
MIN_VALID_OBSERVATIONS = 100

# Capacites des forfaits fictifs
# (vCPU, RAM en GiB)
CATALOG = {
    "S1": (1, 1),
    "S2": (1, 2),
    "S3": (2, 4),
    "S4": (2, 8),
    "S5": (4, 8),
    "S6": (4, 16),
    "S7": (8, 32),
    "S8": (8, 64),
    "S9": (16, 64),
    "S10": (16, 128),
    "S11": (32, 64),
    "S12": (32, 128),
    "S13": (32, 256),
    "S14": (32, 512),
}


def evaluate_plan(group, cores, ram):
    """
    Evalue le risque de depassement des capacites
    reelles d'un forfait sur une periode donnee.

    Risque = proportion d'observations valides
    ou la demande CPU ou RAM depasse la capacite.

    Ce risque est un indicateur experimental :
    il ne mesure pas directement les pannes.
    """

    capacity = pd.to_numeric(
        group["cpu_capacity_mhz"],
        errors="coerce"
    )

    usage = pd.to_numeric(
        group["cpu_usage_mhz"],
        errors="coerce"
    )

    original_cores = pd.to_numeric(
        group["cpu_cores"],
        errors="coerce"
    )

    memory = pd.to_numeric(
        group["mem_used_gb"],
        errors="coerce"
    )

    # Demande CPU en coeurs equivalents
    cpu_demand = (
        original_cores * usage / capacity
    )

    # Observations utilisables
    valid = (
        (capacity > 0)
        & (original_cores > 0)
        & (usage >= 0)
        & (memory >= 0)
        & np.isfinite(cpu_demand)
        & np.isfinite(memory)
    )

    n = int(valid.sum())

    if n == 0:
        return {
            "cpu_risk": np.nan,
            "ram_risk": np.nan,
            "global_risk": np.nan,
            "valid_observations": 0,
            "exceeded_observations": 0,
        }

    cpu_exceeded = (
        (cpu_demand > cores) & valid
    )

    ram_exceeded = (
        (memory > ram) & valid
    )

    any_exceeded = (
        cpu_exceeded | ram_exceeded
    ) & valid

    return {
        "cpu_risk": float(cpu_exceeded.sum() / n),
        "ram_risk": float(ram_exceeded.sum() / n),
        "global_risk": float(any_exceeded.sum() / n),
        "valid_observations": n,
        "exceeded_observations": int(
            any_exceeded.sum()
        ),
    }


def classify_risk(risk, n):
    """
    Classe une configuration selon :
    - la quantite de donnees disponibles ;
    - le seuil de risque experimental.
    """

    if (
        n < MIN_VALID_OBSERVATIONS
        or pd.isna(risk)
    ):
        return "donnees_insuffisantes"

    if risk <= RISK_THRESHOLD:
        return "admissible"

    return "non_admissible"


def main():

    if not BEST_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {BEST_FILE}"
        )

    best = pd.read_csv(
        BEST_FILE,
        dtype={"vm_id": str}
    )

    if best["vm_id"].duplicated().any():
        raise ValueError(
            "Plusieurs forfaits selectionnes "
            "pour une meme VM."
        )

    best_index = best.set_index("vm_id")

    results = []

    for split in ["validation", "test"]:

        path = (
            DATA_DIR
            / f"bitbrains_{split}_v3.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path}"
            )

        df = pd.read_parquet(path)

        df["vm_id"] = (
            df["vm_id"].astype(str)
        )

        for vm_id, group in df.groupby("vm_id"):

            if vm_id not in best_index.index:
                continue

            row = best_index.loc[vm_id]

            plan = row["selected_plan"]

            # Un forfait absent ne doit pas
            # provoquer l'arret de tout le script.
            if (
                pd.isna(plan)
                or plan not in CATALOG
            ):
                results.append({
                    "vm_id": vm_id,
                    "split": split,
                    "selected_method":
                        row["selected_method"],
                    "selected_plan": plan,
                    "plan_cpu": np.nan,
                    "plan_ram_gb": np.nan,
                    "cpu_risk_pct": np.nan,
                    "ram_risk_pct": np.nan,
                    "plan_risk_pct": np.nan,
                    "valid_observations": 0,
                    "exceeded_observations": 0,
                    "status": "forfait_introuvable",
                    "admissible": False,
                })
                continue

            cores, ram = CATALOG[plan]

            metrics = evaluate_plan(
                group,
                cores,
                ram
            )

            risk = metrics["global_risk"]
            n = metrics["valid_observations"]

            status = classify_risk(
                risk,
                n
            )

            results.append({
                "vm_id": vm_id,
                "split": split,
                "selected_method":
                    row["selected_method"],
                "selected_plan": plan,
                "plan_cpu": cores,
                "plan_ram_gb": ram,
                "cpu_risk_pct":
                    metrics["cpu_risk"] * 100,
                "ram_risk_pct":
                    metrics["ram_risk"] * 100,
                "plan_risk_pct":
                    risk * 100,
                "valid_observations": n,
                "exceeded_observations":
                    metrics["exceeded_observations"],
                "status": status,
                "admissible": (
                    status == "admissible"
                ),
            })

    result = pd.DataFrame(results)

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n===== RISQUE DES FORFAITS V3 ====="
    )

    print(
        "Seuil de risque :",
        f"{RISK_THRESHOLD * 100:.1f} %"
    )

    print(
        "Minimum observations valides :",
        MIN_VALID_OBSERVATIONS
    )

    for split, group in result.groupby("split"):

        print(f"\n===== {split.upper()} =====")

        print(
            "VM evaluees :",
            group["vm_id"].nunique()
        )

        counts = group["status"].value_counts()

        for status in [
            "admissible",
            "non_admissible",
            "donnees_insuffisantes",
            "forfait_introuvable",
        ]:
            print(
                f"{status} :",
                int(counts.get(status, 0))
            )

        print(
            "Risque moyen observe (%) :",
            round(
                group["plan_risk_pct"].mean(),
                3
            )
        )

    print("\n===== VM 1, 10, 17 ET 312 =====")

    selected = result[
        (result["split"] == "test")
        & result["vm_id"].isin(
            ["1", "10", "17", "312"]
        )
    ]

    print(
        selected[
            [
                "vm_id",
                "selected_plan",
                "plan_risk_pct",
                "valid_observations",
                "status",
                "admissible",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nFichier cree :", OUTPUT_FILE)


if __name__ == "__main__":
    main()
