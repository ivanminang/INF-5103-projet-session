
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# CLOUDCOST ANALYZER V3 - BACKTEST
# 1249 VM - VALIDATION ET TEST FINAL
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

RECOMMENDATIONS_FILE = DATA_DIR / "vm_recommendations_v3.csv"

DATASETS = {
    "validation": DATA_DIR / "bitbrains_validation_v3.parquet",
    "test": DATA_DIR / "bitbrains_test_v3.parquet",
}

RISK_THRESHOLD = 0.01  # 1 %


def evaluate_configuration(group, cpu_cores, ram_gb):
    """
    Évalue le risque de dépassement CPU ou RAM.

    CPU : conversion des MHz consommés en cœurs
    équivalents, à partir de la capacité provisionnée.

    RAM : utilisation réelle en GiB.
    """

    cpu_capacity = pd.to_numeric(
        group["cpu_capacity_mhz"], errors="coerce"
    )

    cpu_usage = pd.to_numeric(
        group["cpu_usage_mhz"], errors="coerce"
    )

    original_cores = pd.to_numeric(
        group["cpu_cores"], errors="coerce"
    )

    ram_usage = pd.to_numeric(
        group["mem_used_gb"], errors="coerce"
    )

    # Conversion de la consommation CPU en
    # nombre de cœurs équivalents.
    cpu_demand = (
        original_cores * cpu_usage / cpu_capacity
    )

    # Les valeurs CPU invalides ne sont pas utilisées.
    cpu_demand = cpu_demand.where(
        (cpu_capacity > 0)
        & (original_cores > 0)
        & (cpu_usage >= 0)
        & np.isfinite(cpu_demand)
    )

    ram_demand = ram_usage.where(
        (ram_usage >= 0)
        & np.isfinite(ram_usage)
    )

    both_valid = (
        cpu_demand.notna() & ram_demand.notna()
    )

    valid_count = int(both_valid.sum())

    if valid_count == 0:
        return {
            "cpu_risk": np.nan,
            "ram_risk": np.nan,
            "global_risk": np.nan,
            "valid_observations": 0,
            "exceeded_observations": 0,
        }

    cpu_exceeded = (
        cpu_demand > float(cpu_cores)
    ) & both_valid

    ram_exceeded = (
        ram_demand > float(ram_gb)
    ) & both_valid

    global_exceeded = (
        cpu_exceeded | ram_exceeded
    )

    return {
        "cpu_risk": cpu_exceeded.sum() / valid_count,
        "ram_risk": ram_exceeded.sum() / valid_count,
        "global_risk": global_exceeded.sum() / valid_count,
        "valid_observations": valid_count,
        "exceeded_observations": int(
            global_exceeded.sum()
        ),
    }


def run_backtest(data, recommendations):
    """Évalue les trois configurations pour chaque VM."""

    results = []

    data = data.copy()
    data["vm_id"] = data["vm_id"].astype(str)

    recommendations = recommendations.copy()
    recommendations["vm_id"] = (
        recommendations["vm_id"].astype(str)
    )

    rec_by_vm = recommendations.set_index("vm_id")

    for vm_id, group in data.groupby("vm_id"):

        if vm_id not in rec_by_vm.index:
            continue

        rec = rec_by_vm.loc[vm_id]

        configurations = {
            "current": (
                rec["current_cpu_cores"],
                rec["current_ram_gb"],
            ),
            "A": (
                rec["a_cpu_cores"],
                rec["a_ram_gb"],
            ),
            "B": (
                rec["b_cpu_cores"],
                rec["b_ram_gb"],
            ),
        }

        for method, (cores, ram) in configurations.items():

            metrics = evaluate_configuration(
                group, cores, ram
            )

            risk = metrics["global_risk"]

            results.append({
                "vm_id": vm_id,
                "method": method,
                "cpu_cores": cores,
                "ram_gb": ram,
                **metrics,
                "admissible": (
                    pd.notna(risk)
                    and risk <= RISK_THRESHOLD
                ),
            })

    return pd.DataFrame(results)


def summarize(results):
    """Résumé statistique par méthode."""

    summary = results.groupby("method").agg(
        risk_mean=("global_risk", "mean"),
        admissible_count=("admissible", "sum"),
        configurations=("vm_id", "count"),
        observations_valides=(
            "valid_observations", "sum"
        ),
    )

    summary["risk_mean_pct"] = (
        summary["risk_mean"] * 100
    )

    return summary[
        [
            "risk_mean_pct",
            "admissible_count",
            "configurations",
            "observations_valides",
        ]
    ]


def main():

    if not RECOMMENDATIONS_FILE.exists():
        raise FileNotFoundError(
            RECOMMENDATIONS_FILE
        )

    recommendations = pd.read_csv(
        RECOMMENDATIONS_FILE,
        dtype={"vm_id": str}
    )

    print("\n===== BACKTEST CLOUDCOST V3 =====")
    print(
        "VM avec recommandations :",
        len(recommendations)
    )

    for dataset_name, file_path in DATASETS.items():

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        print(
            f"\n===== {dataset_name.upper()} ====="
        )

        data = pd.read_parquet(file_path)

        results = run_backtest(
            data, recommendations
        )

        output_file = (
            DATA_DIR
            / f"vm_backtest_{dataset_name}_v3.csv"
        )

        results.to_csv(
            output_file,
            index=False
        )

        print(
            "VM évaluées :",
            results["vm_id"].nunique()
        )

        print("\nRésumé par méthode :")
        print(
            summarize(results)
            .round(3)
            .to_string()
        )

        print("\nDétails VM 1, 10 et 17 :")

        selected = results[
            results["vm_id"].isin(
                ["1", "10", "17"]
            )
        ].copy()

        selected["risk_pct"] = (
            selected["global_risk"] * 100
        )

        print(
            selected[
                [
                    "vm_id",
                    "method",
                    "cpu_cores",
                    "ram_gb",
                    "risk_pct",
                    "valid_observations",
                    "admissible",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )

        print("\nFichier créé :", output_file)


if __name__ == "__main__":
    main()
