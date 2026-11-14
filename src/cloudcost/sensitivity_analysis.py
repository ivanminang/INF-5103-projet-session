
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# CLOUDCOST ANALYZER V3 - ANALYSE DE SENSIBILITE
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

RECOMMENDATIONS_FILE = (
    DATA_DIR / "vm_recommendations_v3.csv"
)
CANDIDATES_FILE = (
    DATA_DIR / "vm_plan_candidates_v3.csv"
)

SUMMARY_FILE = DATA_DIR / "sensitivity_results_v3.csv"
DETAIL_FILE = DATA_DIR / "sensitivity_details_v3.csv"

RISK_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]
SAFETY_MARGINS = [0.00, 0.15, 0.30]
MIN_OBSERVATIONS = 100


def main():
    rec = pd.read_csv(
        RECOMMENDATIONS_FILE,
        dtype={"vm_id": str}
    )

    candidates = pd.read_csv(
        CANDIDATES_FILE,
        dtype={"vm_id": str}
    )

    required_rec = [
        "vm_id", "current_cpu_cores",
        "current_ram_gb", "cpu_p95_train",
        "ram_p95_train"
    ]

    required_candidates = [
        "vm_id", "plan", "plan_cpu",
        "plan_ram_gb", "monthly_cost",
        "current_plan", "current_cost",
        "plan_risk_pct", "valid_observations"
    ]

    for name, df, required in [
        ("recommandations", rec, required_rec),
        ("candidats", candidates, required_candidates)
    ]:
        missing = [c for c in required if c not in df]
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans {name}: {missing}"
            )

    if rec["vm_id"].duplicated().any():
        raise ValueError("VM dupliquees dans recommandations.")

    candidates["plan_risk_pct"] = pd.to_numeric(
        candidates["plan_risk_pct"], errors="coerce"
    )

    rec_index = rec.set_index("vm_id")
    groups = dict(tuple(candidates.groupby("vm_id")))

    summaries = []
    details = []

    for margin in SAFETY_MARGINS:
        for threshold in RISK_THRESHOLDS:

            scenario = []

            for vm_id, plans in groups.items():
                if vm_id not in rec_index.index:
                    continue

                vm = rec_index.loc[vm_id]

                current_cpu = float(
                    vm["current_cpu_cores"]
                )
                current_ram = float(
                    vm["current_ram_gb"]
                )

                cpu_p95 = vm["cpu_p95_train"]
                ram_p95 = vm["ram_p95_train"]

                # Si un percentile est absent,
                # conserver la capacite actuelle.
                if pd.isna(cpu_p95):
                    required_cpu = current_cpu
                else:
                    required_cpu = min(
                        current_cpu,
                        max(
                            1,
                            np.ceil(
                                current_cpu
                                * cpu_p95 / 100
                                * (1 + margin)
                            )
                        )
                    )

                if pd.isna(ram_p95):
                    required_ram = current_ram
                else:
                    required_ram = min(
                        current_ram,
                        max(
                            1,
                            np.ceil(
                                current_ram
                                * ram_p95 / 100
                                * (1 + margin)
                            )
                        )
                    )

                current_plan = plans[
                    plans["plan"]
                    == plans["current_plan"].iloc[0]
                ]

                if current_plan.empty:
                    raise ValueError(
                        f"Forfait actuel absent: VM {vm_id}"
                    )

                current = current_plan.iloc[0]
                initial_cost = float(
                    current["current_cost"]
                )

                # Filtrer sur la capacite B recalculee,
                # le risque de validation et le prix.
                eligible = plans[
                    (plans["plan_cpu"] >= required_cpu)
                    & (plans["plan_ram_gb"] >= required_ram)
                    & (
                        plans["valid_observations"]
                        >= MIN_OBSERVATIONS
                    )
                    & (
                        plans["plan_risk_pct"]
                        <= threshold * 100
                    )
                    & (
                        plans["monthly_cost"]
                        < initial_cost
                    )
                ].copy()

                if eligible.empty:
                    selected = current
                    optimized = False
                else:
                    selected = eligible.sort_values(
                        ["monthly_cost", "plan_risk_pct"]
                    ).iloc[0]
                    optimized = True

                selected_risk = selected["plan_risk_pct"]
                selected_n = int(
                    selected["valid_observations"]
                )

                admissible = bool(
                    selected_n >= MIN_OBSERVATIONS
                    and pd.notna(selected_risk)
                    and selected_risk <= threshold * 100
                )

                optimized_cost = float(
                    selected["monthly_cost"]
                )

                scenario.append({
                    "vm_id": vm_id,
                    "margin_pct": margin * 100,
                    "threshold_pct": threshold * 100,
                    "required_cpu": required_cpu,
                    "required_ram_gb": required_ram,
                    "selected_plan": selected["plan"],
                    "optimized": optimized,
                    "admissible_validation": admissible,
                    "validation_risk_pct": selected_risk,
                    "current_cost": initial_cost,
                    "optimized_cost": optimized_cost,
                    "monthly_savings": (
                        initial_cost - optimized_cost
                    )
                })

            detail = pd.DataFrame(scenario)

            if detail.empty:
                continue

            initial_total = detail["current_cost"].sum()
            optimized_total = detail["optimized_cost"].sum()
            savings = initial_total - optimized_total

            summaries.append({
                "margin_pct": margin * 100,
                "threshold_pct": threshold * 100,
                "vm_count": len(detail),
                "optimized_vm": int(
                    detail["optimized"].sum()
                ),
                "admissible_validation": int(
                    detail["admissible_validation"].sum()
                ),
                "current_cost": initial_total,
                "optimized_cost": optimized_total,
                "monthly_savings": savings,
                "savings_pct": (
                    100 * savings / initial_total
                    if initial_total > 0 else np.nan
                )
            })

            details.extend(scenario)

    summary = pd.DataFrame(summaries)
    detail = pd.DataFrame(details)

    summary.to_csv(SUMMARY_FILE, index=False)
    detail.to_csv(DETAIL_FILE, index=False)

    print("\n===== SENSIBILITE CLOUDCOST V3 =====")
    print("Scenarios :", len(summary))

    print("\nResultats :")
    print(
        summary[[
            "margin_pct",
            "threshold_pct",
            "vm_count",
            "optimized_vm",
            "admissible_validation",
            "monthly_savings",
            "savings_pct"
        ]]
        .round(2)
        .to_string(index=False)
    )

    print("\nFichiers crees :")
    print(SUMMARY_FILE)
    print(DETAIL_FILE)


if __name__ == "__main__":
    main()
