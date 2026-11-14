
from pathlib import Path
import pandas as pd
import numpy as np


# ==========================================================
# CLOUDCOST ANALYZER - INGESTION BITBRAINS
# ==========================================================

# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw" / "bitbrains"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Colonnes originales Bitbrains
REQUIRED_COLUMNS = [
    "Timestamp [ms]",
    "CPU cores",
    "CPU capacity provisioned [MHZ]",
    "CPU usage [MHZ]",
    "CPU usage [%]",
    "Memory capacity provisioned [KB]",
    "Memory usage [KB]",
]


def read_bitbrains_file(file_path):
    """
    Lit un fichier CSV Bitbrains et normalise les données.
    Chaque fichier représente une machine virtuelle.
    """

    df = pd.read_csv(
        file_path,
        sep=r";\s*",
        engine="python"
    )

    df.columns = df.columns.str.strip()

    # Vérifier les colonnes
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {file_path.name}: "
            f"{missing}"
        )

    # Conversion numérique
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Création du schéma normalisé
    # Conserver le même index que les données originales
    result = pd.DataFrame(index=df.index)

    # Identifier la VM à partir du nom du fichier
    result["vm_id"] = file_path.stem

    # Malgré le nom [ms], les timestamps observés
    # sont exprimés en secondes Unix.
    result["timestamp"] = pd.to_datetime(
        df["Timestamp [ms]"],
        unit="s",
        utc=True,
        errors="coerce"
    )

    result["cpu_cores"] = df["CPU cores"]

    result["cpu_capacity_mhz"] = df[
        "CPU capacity provisioned [MHZ]"
    ]

    result["cpu_usage_mhz"] = df[
        "CPU usage [MHZ]"
    ]

    result["cpu_pct"] = df["CPU usage [%]"]

    result["mem_alloc_gb"] = (
        df["Memory capacity provisioned [KB]"]
        / (1024 ** 2)
    )

    result["mem_used_gb"] = (
        df["Memory usage [KB]"]
        / (1024 ** 2)
    )

    # Pourcentage de RAM consommée
    result["mem_pct"] = np.where(
        result["mem_alloc_gb"] > 0,
        100 * result["mem_used_gb"]
        / result["mem_alloc_gb"],
        np.nan
    )

    result["source"] = "bitbrains_fast"

    # Supprimer les lignes sans timestamp
    result = result.dropna(subset=["timestamp"])

    # Supprimer les lignes dont la capacité est invalide
    result = result[
        (result["cpu_cores"] > 0)
        & (result["cpu_capacity_mhz"] > 0)
        & (result["mem_alloc_gb"] > 0)
    ].copy()

    # Conserver les anomalies CPU/RAM pour audit.
    # Ne pas masquer les pics par un clip automatique.
    result["cpu_out_of_range"] = (
        result["cpu_pct"].notna()
        & ~result["cpu_pct"].between(0, 100)
    )


    # Détecter les anomalies RAM
    result["mem_out_of_range"] = (
    result["mem_pct"].notna()
    & ~result["mem_pct"].between(0, 100)
    )

    # Sauvegarder la valeur originale pour audit
    result["mem_pct_original"] = result["mem_pct"]

    # Exclure les valeurs RAM invalides des analyses
    result.loc[
    result["mem_out_of_range"],
    ["mem_pct", "mem_used_gb"]
    ] = np.nan















    # Conserver les anomalies CPU/RAM pour audit.
    # Ne pas masquer les pics par un clip automatique.
    # result["cpu_out_of_range"] = (
    #     result["cpu_pct"].notna()
    #     & ~result["cpu_pct"].between(0, 100)
    # )

    # result["mem_out_of_range"] = (
    #     result["mem_pct"].notna()
    #     & ~result["mem_pct"].between(0, 100)
    # # Sauvegarder la valeur originale pour audit
    # result["mem_pct_original"] = result["mem_pct"]

    # # Exclure les valeurs RAM invalides des analyses
    # result.loc[
    # result["mem_out_of_range"],
    # ["mem_pct", "mem_used_gb"]
    # ] = np.nan
    # )









    # Éviter les doublons temporels
    result = result.drop_duplicates(
        subset=["vm_id", "timestamp"]
    )

    result = result.sort_values("timestamp")

    return result.reset_index(drop=True)


def load_bitbrains(limit=None):
    """
    Charge plusieurs fichiers Bitbrains.

    limit=50   : premières 50 VM
    limit=None : toutes les VM
    """

    files = sorted(
        RAW_DIR.glob("*.csv"),
        key=lambda p: (
            0, int(p.stem)
        ) if p.stem.isdigit() else (
            1, p.stem
        )
    )

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier CSV trouvé dans {RAW_DIR}"
        )

    if limit is not None:
        files = files[:limit]

    datasets = []
    errors = []

    print(f"Fichiers sélectionnés : {len(files)}")

    for i, file_path in enumerate(files, start=1):
        try:
            df = read_bitbrains_file(file_path)
            if df.empty:
                raise ValueError("Aucune ligne valide")

            datasets.append(df)

            print(
                f"[{i}/{len(files)}] "
                f"{file_path.name}: {len(df)} lignes"
            )

        except Exception as e:
            errors.append({
                "file": file_path.name,
                "error": str(e)
            })
            print(f"ERREUR {file_path.name}: {e}")

    if not datasets:
        raise ValueError(
            "Aucune VM n'a pu être chargée."
        )

    combined = pd.concat(
        datasets, ignore_index=True
    )

    return combined, pd.DataFrame(errors)


def main():
    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    # PHASE 1 : 50 VM
    # PHASE 2 : remplacer par None pour tout traiter
    limit = None

    df, errors = load_bitbrains(limit=limit)

    # Export des données normalisées
    output_file = (
        OUTPUT_DIR / "bitbrains_normalized.parquet"
    )

    df.to_parquet(
        output_file,
        index=False
    )

    # Rapport d'erreurs
    errors.to_csv(
        OUTPUT_DIR / "ingest_errors.csv",
        index=False
    )

    print("\n========== RÉSULTATS ==========")
    print("VM chargées :", df["vm_id"].nunique())
    print("Observations :", len(df))
    print("Début :", df["timestamp"].min())
    print("Fin :", df["timestamp"].max())

    print("\nValeurs manquantes :")
    print(
        df[["cpu_pct", "mem_pct"]]
        .isna().sum()
    )

    print("\nValeurs hors plage :")
    print(
        df[
            ["cpu_out_of_range", "mem_out_of_range"]
        ].sum()
    )

    print("\nFichier créé :", output_file)
    print("Fichiers en erreur :", len(errors))


if __name__ == "__main__":
    main()
