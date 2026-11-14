
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# CLOUDCOST ANALYZER V3
# RECOMMANDATIONS SUR APPRENTISSAGE UNIQUEMENT
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "bitbrains_train_v3.parquet"
OUTPUT_FILE = DATA_DIR / "vm_recommendations_v3.csv"

CPU_THRESHOLD_A = 20.0
SAFETY_MARGIN = 0.15


def method_a(group):
    """
    Méthode A :
    Si la moyenne CPU est inférieure à 20 %,
    réduire théoriquement les cœurs de moitié.

    La RAM reste inchangée.
    """

    current_cores = int(
        np.ceil(group["cpu_cores"].median())
    )

    current_ram = float(
        group["mem_alloc_gb"].median()
    )

    cpu_mean = float(
        group["cpu_pct"].mean()
    )

    if pd.isna(cpu_mean):
        recommended_cores = current_cores

    elif cpu_mean < CPU_THRESHOLD_A:
        recommended_cores = max(
            1,
            int(np.ceil(current_cores / 2))
        )

    else:
        recommended_cores = current_cores

    return {
        "cpu_mean_train": cpu_mean,
        "a_cpu_cores": recommended_cores,
        "a_ram_gb": current_ram,
    }



def method_b(group):
    """
    Méthode B : P95 de la demande CPU en cœurs
    équivalents et de la RAM utilisée, avec
    une marge de sécurité de 15 %.
    """

    current_cores = int(
        np.ceil(group["cpu_cores"].median())
    )

    current_ram = float(
        group["mem_alloc_gb"].median()
    )

    # Demande CPU en cœurs équivalents.
    # Le ratio MHz est préférable au champ CPU usage [%]
    # pour cette estimation de capacité.
    cpu_demand = (
        group["cpu_cores"]
        * group["cpu_usage_mhz"]
        / group["cpu_capacity_mhz"]
    )

    cpu_demand = cpu_demand.replace(
        [np.inf, -np.inf], np.nan
    )

    cpu_demand = cpu_demand.where(
        cpu_demand >= 0
    )

    cpu_p95_cores = cpu_demand.quantile(0.95)

    # Besoin RAM directement en GiB
    ram_p95_gb = group["mem_used_gb"].quantile(0.95)

    if pd.notna(cpu_p95_cores):
        recommended_cores = max(
            1,
            int(np.ceil(
                cpu_p95_cores * (1 + SAFETY_MARGIN)
            ))
        )
    else:
        recommended_cores = current_cores

    if pd.notna(ram_p95_gb):
        recommended_ram = max(
            1,
            int(np.ceil(
                ram_p95_gb * (1 + SAFETY_MARGIN)
            ))
        )
    else:
        recommended_ram = current_ram

    recommended_cores = min(
        recommended_cores, current_cores
    )

    recommended_ram = min(
        recommended_ram, current_ram
    )

    return {
        "cpu_p95_train": (
            cpu_p95_cores / current_cores * 100
            if pd.notna(cpu_p95_cores)
            else np.nan
        ),
        "ram_p95_train": (
            ram_p95_gb / current_ram * 100
            if pd.notna(ram_p95_gb)
            else np.nan
        ),
        "b_cpu_cores": recommended_cores,
        "b_ram_gb": recommended_ram,
    }


def generate_recommendations(train):
    """
    Génère une recommandation A et B
    pour chaque machine virtuelle.
    """

    results = []

    for vm_id, group in train.groupby("vm_id"):

        if group.empty:
            continue

        current_cores = int(
            np.ceil(group["cpu_cores"].median())
        )

        current_ram = float(
            group["mem_alloc_gb"].median()
        )

        # Vérification des capacités existantes
        if (
            current_cores <= 0
            or not np.isfinite(current_ram)
            or current_ram <= 0
        ):
            print(
                f"VM {vm_id} ignorée : "
                "capacité invalide."
            )
            continue

        row = {
            "vm_id": vm_id,
            "train_observations": len(group),
            "current_cpu_cores": current_cores,
            "current_ram_gb": current_ram,
        }

        row.update(method_a(group))
        row.update(method_b(group))

        results.append(row)

    return pd.DataFrame(results)


def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}\n"
            "Exécute d'abord split_data.py."
        )

    train = pd.read_parquet(INPUT_FILE)

    # Vérifier les colonnes nécessaires




    required_columns = [
    "vm_id",
    "timestamp",
    "cpu_cores",
    "cpu_pct",
    "cpu_capacity_mhz",
    "cpu_usage_mhz",
    "mem_alloc_gb",
    "mem_used_gb",
    "mem_pct",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in train.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    recommendations = generate_recommendations(
        train
    )

    if recommendations.empty:
        raise ValueError(
            "Aucune recommandation générée."
        )

    recommendations.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===== RECOMMANDATIONS V3 =====")

    print(
        "VM analysées :",
        len(recommendations)
    )

    print(
        "Observations apprentissage :",
        len(train)
    )

    print(
        "Méthode A : moyenne CPU < 20 %"
    )

    print(
        "Méthode B : P95 CPU/RAM + 15 %"
    )

    columns = [
        "vm_id",
        "current_cpu_cores",
        "a_cpu_cores",
        "b_cpu_cores",
        "current_ram_gb",
        "b_ram_gb",
    ]

    print("\nAperçu des recommandations :")

    print(
        recommendations[columns]
        .head(15)
        .round(2)
        .to_string(index=False)
    )

    print(
        "\nFichier créé :",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
