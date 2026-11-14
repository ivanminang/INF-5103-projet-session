
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# CLOUDCOST ANALYZER V3 - VISUALISATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR / "data" / "processed"
    / "bitbrains_normalized.parquet"
)

STATS_FILE = (
    BASE_DIR / "data" / "processed"
    / "vm_statistics.csv"
)

OUTPUT_DIR = BASE_DIR / "reports" / "figures"

# VM sélectionnées pour l'analyse
VM_IDS = ["1", "10", "17"]

sns.set_theme(style="whitegrid")


def load_data():
    """Charge les données normalisées et statistiques."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    if not STATS_FILE.exists():
        raise FileNotFoundError(STATS_FILE)

    df = pd.read_parquet(DATA_FILE)
    stats = pd.read_csv(
        STATS_FILE,
        dtype={"vm_id": str}
    )

    df["vm_id"] = df["vm_id"].astype(str)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )

    return df, stats


def plot_timeseries(df, vm_id):
    """Affiche l'évolution CPU et RAM d'une VM."""

    vm = df[df["vm_id"] == vm_id].copy()
    vm = vm.sort_values("timestamp")

    if vm.empty:
        print(f"VM {vm_id} introuvable.")
        return

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 7),
        sharex=True
    )

    axes[0].plot(
        vm["timestamp"],
        vm["cpu_pct"],
        color="steelblue",
        linewidth=0.8
    )
    axes[0].set_title(
        f"VM {vm_id} - Évolution CPU"
    )
    axes[0].set_ylabel("CPU (%)")

    axes[1].plot(
        vm["timestamp"],
        vm["mem_pct"],
        color="darkorange",
        linewidth=0.8
    )
    axes[1].set_title(
        f"VM {vm_id} - Évolution RAM"
    )
    axes[1].set_ylabel("RAM (%)")
    axes[1].set_xlabel("Date (UTC)")

    plt.tight_layout()

    output = OUTPUT_DIR / f"vm_{vm_id}_timeseries.png"
    plt.savefig(output, dpi=150)
    plt.close()

    print("Graphique créé :", output)


def plot_cpu_distribution(df, vm_id):
    """Affiche la distribution CPU d'une VM."""

    vm = df[df["vm_id"] == vm_id]

    cpu = vm["cpu_pct"].dropna()

    if cpu.empty:
        return

    plt.figure(figsize=(10, 5))

    sns.histplot(
        cpu,
        bins=50,
        kde=True,
        color="steelblue"
    )

    plt.axvline(
        cpu.mean(),
        color="red",
        linestyle="--",
        label=f"Moyenne : {cpu.mean():.2f}%"
    )

    plt.axvline(
        cpu.quantile(0.95),
        color="green",
        linestyle="--",
        label=f"P95 : {cpu.quantile(0.95):.2f}%"
    )

    plt.axvline(
        cpu.quantile(0.99),
        color="purple",
        linestyle="--",
        label=f"P99 : {cpu.quantile(0.99):.2f}%"
    )

    plt.title(
        f"VM {vm_id} - Distribution CPU"
    )
    plt.xlabel("Utilisation CPU (%)")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.tight_layout()

    output = OUTPUT_DIR / f"vm_{vm_id}_cpu_distribution.png"
    plt.savefig(output, dpi=150)
    plt.close()

    print("Graphique créé :", output)


def plot_vm_comparison(stats):
    """Compare moyenne, P95 et P99 CPU."""

    selected = stats[
        stats["vm_id"].isin(VM_IDS)
    ].copy()

    if selected.empty:
        return


    selected = selected.set_index("vm_id")
    selected = selected.reindex(VM_IDS)

    
    # selected = selected.set_index("vm_id").loc[
    #     [vm for vm in VM_IDS
    #      if vm in selected["vm_id"].values]
    # ] if False else selected

    # selected = selected.set_index("vm_id")
    # selected = selected.reindex(VM_IDS)



    columns = [
        "cpu_pct_mean",
        "cpu_pct_p95",
        "cpu_pct_p99"
    ]

    ax = selected[columns].plot(
        kind="bar",
        figsize=(10, 6),
        color=[
            "steelblue",
            "darkorange",
            "seagreen"
        ]
    )

    ax.set_title(
        "Comparaison CPU - VM 1, 10 et 17"
    )
    ax.set_xlabel("Machine virtuelle")
    ax.set_ylabel("Utilisation CPU (%)")
    ax.legend(["Moyenne", "P95", "P99"])
    plt.xticks(rotation=0)
    plt.tight_layout()

    output = OUTPUT_DIR / "vm_cpu_comparison.png"
    plt.savefig(output, dpi=150)
    plt.close()

    print("Graphique créé :", output)


def main():

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    df, stats = load_data()

    print("VM disponibles :", df["vm_id"].nunique())

    for vm_id in VM_IDS:
        plot_timeseries(df, vm_id)
        plot_cpu_distribution(df, vm_id)

    plot_vm_comparison(stats)

    print("\nVisualisations terminées.")


if __name__ == "__main__":
    main()
