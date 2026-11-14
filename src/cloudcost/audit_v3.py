
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CLOUDCOST ANALYZER V3
# Audit de coherence des resultats
# Projet INF-5103 - UQO
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

THRESHOLD = 1.0
MIN_OBSERVATIONS = 100
TOLERANCE = 0.01

FILES = {
    "best": "vm_best_recommendations_v3.csv",
    "pricing": "vm_pricing_v3.csv",
    "validation": "vm_plan_validation_v3.csv",
    "risk": "vm_plan_risk_v3.csv",
}

issues = []
checks = []


def load_csv(key):
    path = DATA_DIR / FILES[key]

    if not path.exists():
        print(f"[ABSENT] {path.name}")
        return None

    df = pd.read_csv(path, dtype={"vm_id": str})

    print(f"[OK] {path.name}: {len(df)} lignes")
    return df


def record(vm_id, check, expected, observed, severity="ERROR"):
    issues.append({
        "vm_id": str(vm_id),
        "controle": check,
        "attendu": str(expected),
        "observe": str(observed),
        "gravite": severity,
    })


def to_bool(value):
    if pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if str(value).strip().lower() in ("true", "1", "yes", "oui"):
        return True

    if str(value).strip().lower() in ("false", "0", "no", "non"):
        return False

    return None


def same_number(a, b, tol=TOLERANCE):
    if pd.isna(a) or pd.isna(b):
        return pd.isna(a) and pd.isna(b)

    return bool(np.isclose(
        float(a), float(b),
        atol=tol, rtol=0
    ))


def check_duplicates(df, name, keys):
    if df is None:
        return

    if not set(keys).issubset(df.columns):
        return

    duplicates = df[df.duplicated(keys, keep=False)]

    if not duplicates.empty:
        print(
            f"[ATTENTION] {name}: "
            f"{len(duplicates)} lignes avec cles dupliquees"
        )

        for _, row in duplicates.iterrows():
            record(
                row["vm_id"],
                f"doublon_{name}",
                "Cle unique",
                tuple(row[k] for k in keys)
            )


def compare_column(df, left, right, label, tolerance=TOLERANCE):
    if left not in df.columns or right not in df.columns:
        return

    for _, row in df.iterrows():
        a = row[left]
        b = row[right]

        if not same_number(a, b, tolerance):
            record(
                row["vm_id"],
                label,
                a,
                b
            )


print("\n" + "=" * 60)
print("AUDIT DE COHERENCE - CLOUDCOST ANALYZER V3")
print("=" * 60)

best = load_csv("best")
pricing = load_csv("pricing")
plan_validation = load_csv("validation")
risk = load_csv("risk")

if best is None:
    raise SystemExit(
        "Audit impossible : fichier des recommandations absent."
    )

# ============================================================
# 1. STRUCTURE ET DOUBLONS
# ============================================================

print("\n[1] STRUCTURE ET DOUBLONS")

check_duplicates(best, "best", ["vm_id"])
check_duplicates(
    pricing, "pricing", ["vm_id", "plan"]
)
check_duplicates(
    plan_validation, "plan_validation", ["vm_id"]
)
check_duplicates(
    risk, "risk", ["vm_id", "split"]
)

required_best = [
    "vm_id",
    "selected_plan",
    "current_cost",
    "optimized_cost",
    "monthly_savings",
    "savings_pct",
    "validation_risk_pct",
    "test_risk_pct",
    "admissible_validation",
    "admissible_test",
]

missing = [
    col for col in required_best
    if col not in best.columns
]

if missing:
    raise SystemExit(
        "Colonnes manquantes dans best : "
        + ", ".join(missing)
    )

print(f"VM uniques : {best['vm_id'].nunique()}")

# ============================================================
# 2. COHERENCE DES COUTS
# ============================================================

print("\n[2] COHERENCE DES COUTS")

for _, row in best.iterrows():

    vm = row["vm_id"]

    current = row["current_cost"]
    optimized = row["optimized_cost"]
    savings = row["monthly_savings"]
    savings_pct = row["savings_pct"]

    if pd.isna(current) or pd.isna(optimized):
        record(vm, "cout_manquant", "Cout renseigne", "NaN")
        continue

    expected_savings = current - optimized

    if not same_number(savings, expected_savings):
        record(
            vm,
            "monthly_savings",
            expected_savings,
            savings
        )

    if current > 0:
        expected_pct = 100 * expected_savings / current

        if not same_number(
            savings_pct, expected_pct, tol=0.05
        ):
            record(
                vm,
                "savings_pct",
                expected_pct,
                savings_pct
            )

    if optimized > current + TOLERANCE:
        record(
            vm,
            "cout_superieur_initial",
            f"<= {current}",
            optimized
        )

# ============================================================
# 3. ADMISSIBILITE DES FORFAITS
# ============================================================

print("\n[3] ADMISSIBILITE ET OBSERVATIONS")

for _, row in best.iterrows():

    vm = row["vm_id"]

    for split in ("validation", "test"):

        risk_col = f"{split}_risk_pct"
        flag_col = f"admissible_{split}"
        obs_col = f"{split}_observations"

        value = row[risk_col]
        flag = to_bool(row[flag_col])

        if obs_col in best.columns:
            observations = row[obs_col]
        else:
            observations = np.nan

        if pd.isna(value):
            expected = False
        else:
            enough = (
                not pd.isna(observations)
                and observations >= MIN_OBSERVATIONS
            )

            expected = (
                enough and value <= THRESHOLD + 1e-9
            )

        if flag is not None and flag != expected:
            record(
                vm,
                f"admissibilite_{split}",
                expected,
                flag
            )

# ============================================================
# 4. COMPARAISON BEST / PLAN_RISK
# ============================================================

print("\n[4] COMPARAISON AVEC PLAN_RISK")

if risk is not None:

    needed = [
        "vm_id", "split", "selected_plan",
        "plan_risk_pct", "valid_observations"
    ]

    if set(needed).issubset(risk.columns):

        for split in ("validation", "test"):

            subset = risk[
                risk["split"].astype(str).str.lower() == split
            ].copy()

            merged = best.merge(
                subset,
                on="vm_id",
                how="left",
                suffixes=("_best", "_risk"),
                indicator=True
            )

            for _, row in merged.iterrows():

                vm = row["vm_id"]

                if row["_merge"] == "left_only":
                    record(
                        vm,
                        f"plan_risk_absent_{split}",
                        "Ligne presente",
                        "Absente"
                    )
                    continue

                selected_best = str(
                    row["selected_plan_best"]
                )

                selected_risk = str(
                    row["selected_plan_risk"]
                )

                if selected_best != selected_risk:
                    record(
                        vm,
                        f"forfait_different_{split}",
                        selected_best,
                        selected_risk
                    )

                best_risk = row[f"{split}_risk_pct"]
                measured_risk = row["plan_risk_pct"]

                if not same_number(
                    best_risk, measured_risk
                ):
                    record(
                        vm,
                        f"risque_different_{split}",
                        best_risk,
                        measured_risk
                    )

                obs_col = f"{split}_observations"

                if obs_col in merged.columns:
                    if not same_number(
                        row[obs_col],
                        row["valid_observations"],
                        tol=0
                    ):
                        record(
                            vm,
                            f"observations_differentes_{split}",
                            row[obs_col],
                            row["valid_observations"]
                        )

# ============================================================
# 5. COMPARAISON BEST / PLAN_VALIDATION
# ============================================================

print("\n[5] COMPARAISON AVEC PLAN_VALIDATION")

if plan_validation is not None:

    if "selected_plan" in plan_validation.columns:

        merged = best.merge(
            plan_validation,
            on="vm_id",
            how="left",
            suffixes=("_best", "_control"),
            indicator=True
        )

        for _, row in merged.iterrows():

            vm = row["vm_id"]

            if row["_merge"] == "left_only":
                record(
                    vm,
                    "controle_absent",
                    "Ligne presente",
                    "Absente",
                    "WARNING"
                )
                continue

            selected_best = str(
                row["selected_plan_best"]
            )

            selected_control = str(
                row["selected_plan_control"]
            )

            if selected_best != selected_control:
                record(
                    vm,
                    "forfait_best_vs_controle",
                    selected_best,
                    selected_control,
                    "WARNING"
                )

            if (
                "test_risk_pct_best" in merged.columns
                and "test_risk_pct_control" in merged.columns
            ):
                if not same_number(
                    row["test_risk_pct_best"],
                    row["test_risk_pct_control"]
                ):
                    record(
                        vm,
                        "risque_best_vs_controle",
                        row["test_risk_pct_best"],
                        row["test_risk_pct_control"],
                        "WARNING"
                    )

# ============================================================
# 6. COMPARAISON DES COUTS AU CATALOGUE
# ============================================================

print("\n[6] VERIFICATION DES PRIX CATALOGUE")

if pricing is not None:

    needed = [
        "vm_id", "plan", "monthly_cost"
    ]

    if set(needed).issubset(pricing.columns):

        selected = best[
            best["selected_method"].astype(str)
            .str.lower() == "catalogue"
        ].copy()

        joined = selected.merge(
            pricing[
                ["vm_id", "plan", "monthly_cost"]
            ],
            left_on=["vm_id", "selected_plan"],
            right_on=["vm_id", "plan"],
            how="left"
        )

        for _, row in joined.iterrows():

            vm = row["vm_id"]

            if pd.isna(row["monthly_cost"]):
                record(
                    vm,
                    "prix_catalogue_absent",
                    row["selected_plan"],
                    "Prix introuvable",
                    "WARNING"
                )

            elif not same_number(
                row["optimized_cost"],
                row["monthly_cost"]
            ):
                record(
                    vm,
                    "prix_catalogue_different",
                    row["monthly_cost"],
                    row["optimized_cost"]
                )

# ============================================================
# 7. SYNTHESE ET EXPORT
# ============================================================

print("\n" + "=" * 60)
print("SYNTHESE")
print("=" * 60)

current_total = best["current_cost"].sum()
optimized_total = best["optimized_cost"].sum()
savings_total = current_total - optimized_total

print(f"VM analysees : {len(best)}")
print(f"Cout initial : {current_total:,.2f} USD/mois")
print(f"Cout optimise : {optimized_total:,.2f} USD/mois")
print(f"Economies : {savings_total:,.2f} USD/mois")

for split in ("validation", "test"):
    col = f"admissible_{split}"

    flags = best[col].map(to_bool)
    admissible_count = int((flags == True).sum())

    print(
        f"Admissibles {split} : "
        f"{admissible_count}/{len(best)}"
    )

if issues:

    audit = pd.DataFrame(issues)

    print(f"\nAnomalies/signaux : {len(audit)}")

    print("\nRepartition par controle :")
    print(
        audit["controle"]
        .value_counts()
        .to_string()
    )

    print("\nExemples :")
    print(
        audit.head(20).to_string(index=False)
    )

else:
    audit = pd.DataFrame(
        columns=[
            "vm_id", "controle", "attendu",
            "observe", "gravite"
        ]
    )

    print("\nAucune incoherence detectee.")

output = DATA_DIR / "audit_coherence_v3.csv"

audit.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nRapport exporte : {output}")
print("\nAUDIT TERMINE")
