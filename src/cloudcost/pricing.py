
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# CLOUDCOST ANALYZER V3 - PRICING CORRIGE
# Selection sur validation, evaluation sur test final
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

CANDIDATES_FILE = DATA_DIR / "vm_plan_candidates_v3.csv"
TEST_FILE = DATA_DIR / "bitbrains_test_v3.parquet"

BEST_FILE = DATA_DIR / "vm_best_recommendations_v3.csv"
OUTPUT_FILE = DATA_DIR / "vm_pricing_v3.csv"

RISK_THRESHOLD = 0.01
MIN_VALID_OBSERVATIONS = 100


def evaluate_plan(group, cores, ram):
    """Risque de dépassement CPU ou RAM."""

    capacity = pd.to_numeric(
        group["cpu_capacity_mhz"], errors="coerce"
    )
    usage = pd.to_numeric(
        group["cpu_usage_mhz"], errors="coerce"
    )
    original_cores = pd.to_numeric(
        group["cpu_cores"], errors="coerce"
    )
    memory = pd.to_numeric(
        group["mem_used_gb"], errors="coerce"
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        cpu_demand = original_cores * usage / capacity

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
        (cpu_demand > cores)
        | (memory > ram)
    ) & valid

    return float(exceeded.sum() / n), n


def main():

    candidates = pd.read_csv(
        CANDIDATES_FILE,
        dtype={"vm_id": str}
    )

    test = pd.read_parquet(TEST_FILE)
    test["vm_id"] = test["vm_id"].astype(str)

    test_groups = {
        vm_id: group
        for vm_id, group in test.groupby("vm_id")
    }

    results = []

    for vm_id, group in candidates.groupby("vm_id"):

        current_plan = group["current_plan"].iloc[0]
        current_cost = float(group["current_cost"].iloc[0])

        # Chercher le forfait actuel dans les candidats.
        current_rows = group[
            group["plan"] == current_plan
        ]

        if current_rows.empty:
            raise ValueError(
                f"Forfait actuel absent pour VM {vm_id}"
            )

        current = current_rows.iloc[0]

        # Seules les donnees de validation
        # servent a selectionner le forfait.
        admissible = group[
            (group["admissible"] == True)
            & (group["monthly_cost"] < current_cost)
        ].copy()

        if admissible.empty:
            selected = current
            method = "current"
        else:
            selected = admissible.sort_values(
                ["monthly_cost", "plan_risk_pct"]
            ).iloc[0]
            method = "catalogue"

        selected_plan = selected["plan"]
        selected_cost = float(selected["monthly_cost"])
        selected_cpu = float(selected["plan_cpu"])
        selected_ram = float(selected["plan_ram_gb"])

        # Test final : evaluation seulement.
        test_group = test_groups.get(vm_id)

        if test_group is None:
            test_risk = np.nan
            test_n = 0
        else:
            test_risk, test_n = evaluate_plan(
                test_group,
                selected_cpu,
                selected_ram
            )

        validation_risk_pct = selected["plan_risk_pct"]

        validation_ok = bool(
            selected["admissible"]
        )

        test_ok = bool(
            test_n >= MIN_VALID_OBSERVATIONS
            and pd.notna(test_risk)
            and test_risk <= RISK_THRESHOLD
        )

        savings = current_cost - selected_cost

        results.append({
            "vm_id": vm_id,
            "selected_method": method,
            "selected_plan": selected_plan,
            "current_plan": current_plan,
            "current_cost": current_cost,
            "optimized_cost": selected_cost,
            "monthly_savings": savings,
            "savings_pct": (
                100 * savings / current_cost
                if current_cost > 0 else np.nan
            ),
            "plan_cpu": selected_cpu,
            "plan_ram_gb": selected_ram,
            "validation_risk_pct": validation_risk_pct,
            "validation_observations": int(
                selected["valid_observations"]
            ),
            "test_risk_pct": (
                100 * test_risk
                if pd.notna(test_risk)
                else np.nan
            ),
            "test_observations": test_n,
            "admissible_validation": validation_ok,
            "admissible_test": test_ok,
        })

    best = pd.DataFrame(results)

    best.to_csv(BEST_FILE, index=False)
    candidates.to_csv(OUTPUT_FILE, index=False)

    total_initial = best["current_cost"].sum()
    total_optimized = best["optimized_cost"].sum()
    total_savings = best["monthly_savings"].sum()

    print("\n===== PRICING V3 CORRIGE =====")
    print("VM analysees :", len(best))

    print("\n===== ECONOMIES SIMULEES =====")
    print("Cout initial :", round(total_initial, 2))
    print("Cout optimise :", round(total_optimized, 2))
    print("Economies mensuelles :", round(total_savings, 2))

    print(
        "Economies (%) :",
        round(100 * total_savings / total_initial, 2)
        if total_initial > 0 else np.nan
    )

    print(
        "VM avec economies positives :",
        int((best["monthly_savings"] > 0).sum())
    )

    print(
        "VM admissibles sur validation :",
        int(best["admissible_validation"].sum())
    )

    print(
        "VM admissibles sur test final :",
        int(best["admissible_test"].sum())
    )

    print("\nMethodes selectionnees :")
    print(best["selected_method"].value_counts())

    print("\nVM 1, 10 et 17 :")
    print(
        best[best["vm_id"].isin(["1", "10", "17"])]
        [[
            "vm_id",
            "selected_plan",
            "current_cost",
            "optimized_cost",
            "monthly_savings",
            "validation_risk_pct",
            "test_risk_pct",
            "admissible_test",
        ]]
        .round(3)
        .to_string(index=False)
    )

    print("\nFichiers crees :")
    print(BEST_FILE)
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
