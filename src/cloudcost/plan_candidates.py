
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# CLOUDCOST ANALYZER V3
# EVALUATION DES FORFAITS CANDIDATS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

VALIDATION_FILE = (
    DATA_DIR / "bitbrains_validation_v3.parquet"
)

RECOMMENDATIONS_FILE = (
    DATA_DIR / "vm_recommendations_v3.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "vm_plan_candidates_v3.csv"
)

RISK_THRESHOLD = 0.01
MIN_VALID_OBSERVATIONS = 100

# Catalogue fictif : USD/mois
CATALOG = pd.DataFrame([
    {"plan": "S1", "vcpu": 1, "ram_gb": 1, "price": 12},
    {"plan": "S2", "vcpu": 1, "ram_gb": 2, "price": 18},
    {"plan": "S3", "vcpu": 2, "ram_gb": 4, "price": 35},
    {"plan": "S4", "vcpu": 2, "ram_gb": 8, "price": 55},
    {"plan": "S5", "vcpu": 4, "ram_gb": 8, "price": 75},
    {"plan": "S6", "vcpu": 4, "ram_gb": 16, "price": 110},
    {"plan": "S7", "vcpu": 8, "ram_gb": 32, "price": 210},
    {"plan": "S8", "vcpu": 8, "ram_gb": 64, "price": 310},
    {"plan": "S9", "vcpu": 16, "ram_gb": 64, "price": 420},
    {"plan": "S10", "vcpu": 16, "ram_gb": 128, "price": 650},
    {"plan": "S11", "vcpu": 32, "ram_gb": 64, "price": 750},
    {"plan": "S12", "vcpu": 32, "ram_gb": 128, "price": 950},
    {"plan": "S13", "vcpu": 32, "ram_gb": 256, "price": 1350},
    {"plan": "S14", "vcpu": 32, "ram_gb": 512, "price": 2100},
])


def evaluate_plan(group, plan_cpu, plan_ram):
    """
    Calcule le risque observé de dépassement
    de capacité du forfait sur la validation.

    CPU en coeurs équivalents :
    cpu_cores * cpu_usage_mhz / cpu_capacity_mhz

    RAM en GiB :
    mem_used_gb

    Les observations invalides sont exclues.
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

    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):
        cpu_demand = (
            original_cores * usage / capacity
        )

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
        return np.nan, 0

    exceeded = (
        (cpu_demand > plan_cpu)
        | (memory > plan_ram)
    ) & valid

    risk = float(exceeded.sum() / n)

    return risk, n


def main():

    for path in [
        VALIDATION_FILE,
        RECOMMENDATIONS_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    validation = pd.read_parquet(
        VALIDATION_FILE
    )

    recommendations = pd.read_csv(
        RECOMMENDATIONS_FILE,
        dtype={"vm_id": str}
    )

    validation["vm_id"] = (
        validation["vm_id"].astype(str)
    )

    recommendations["vm_id"] = (
        recommendations["vm_id"].astype(str)
    )

    recommendations = recommendations.set_index(
        "vm_id"
    )

    results = []

    for vm_id, group in validation.groupby(
        "vm_id"
    ):

        if vm_id not in recommendations.index:
            continue

        rec = recommendations.loc[vm_id]

        current_cpu = float(
            rec["current_cpu_cores"]
        )

        current_ram = float(
            rec["current_ram_gb"]
        )

        # Forfait actuel de référence :
        # le moins cher couvrant la capacité actuelle.
        current_compatible = CATALOG[
            (CATALOG["vcpu"] >= current_cpu)
            & (CATALOG["ram_gb"] >= current_ram)
        ]

        if current_compatible.empty:
            continue

        current_plan = current_compatible.sort_values(
            ["price", "vcpu", "ram_gb"]
        ).iloc[0]

        for _, plan in CATALOG.iterrows():

            # Ne pas considérer un forfait plus cher
            # que le forfait actuel.
            if plan["price"] > current_plan["price"]:
                continue

            risk, n = evaluate_plan(
                group,
                plan["vcpu"],
                plan["ram_gb"]
            )

            enough_data = (
                n >= MIN_VALID_OBSERVATIONS
            )

            admissible = (
                enough_data
                and pd.notna(risk)
                and risk <= RISK_THRESHOLD
            )

            results.append({
                "vm_id": vm_id,
                "plan": plan["plan"],
                "plan_cpu": plan["vcpu"],
                "plan_ram_gb": plan["ram_gb"],
                "monthly_cost": plan["price"],
                "current_plan": current_plan["plan"],
                "current_cost": current_plan["price"],
                "plan_risk_pct": (
                    risk * 100
                    if pd.notna(risk)
                    else np.nan
                ),
                "valid_observations": n,
                "enough_data": enough_data,
                "admissible": admissible,
            })

    result = pd.DataFrame(results)

    if result.empty:
        raise ValueError(
            "Aucun forfait candidat évalué."
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n===== FORFAITS CANDIDATS V3 ====="
    )

    print(
        "VM évaluées :",
        result["vm_id"].nunique()
    )

    print(
        "Configurations évaluées :",
        len(result)
    )

    print(
        "Configurations admissibles :",
        int(result["admissible"].sum())
    )

    print(
        "VM avec au moins un forfait admissible :",
        result.loc[
            result["admissible"], "vm_id"
        ].nunique()
    )

    print(
        "\nExemple : forfaits de la VM 1"
    )

    columns = [
        "vm_id",
        "plan",
        "plan_cpu",
        "plan_ram_gb",
        "monthly_cost",
        "plan_risk_pct",
        "valid_observations",
        "admissible",
    ]

    print(
        result[result["vm_id"] == "1"][columns]
        .sort_values("monthly_cost")
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nFichier créé :",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
