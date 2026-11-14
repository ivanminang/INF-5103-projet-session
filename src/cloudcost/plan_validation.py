
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CLOUDCOST ANALYZER V3
# CONTROLE DE COHERENCE DES FORFAITS SELECTIONNES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

BEST_FILE = DATA_DIR / "vm_best_recommendations_v3.csv"
RISK_FILE = DATA_DIR / "vm_plan_risk_v3.csv"
PRICING_FILE = DATA_DIR / "vm_pricing_v3.csv"

OUTPUT_FILE = DATA_DIR / "vm_plan_validation_v3.csv"

RISK_THRESHOLD_PCT = 1.0
MIN_VALID_OBSERVATIONS = 100
TOLERANCE = 0.01

# Catalogue fictif : CPU, RAM (GiB), USD/mois
CATALOG = {
    "S1": (1, 1, 12),
    "S2": (1, 2, 18),
    "S3": (2, 4, 35),
    "S4": (2, 8, 55),
    "S5": (4, 8, 75),
    "S6": (4, 16, 110),
    "S7": (8, 32, 210),
    "S8": (8, 64, 310),
    "S9": (16, 64, 420),
    "S10": (16, 128, 650),
    "S11": (32, 64, 750),
    "S12": (32, 128, 950),
    "S13": (32, 256, 1350),
    "S14": (32, 512, 2100),
}


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    return pd.read_csv(
        path,
        dtype={"vm_id": str}
    )


def as_number(value):
    return pd.to_numeric(
        pd.Series([value]),
        errors="coerce"
    ).iloc[0]


def as_bool(value):
    if pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = str(value).strip().lower()

    if text in ("true", "1", "yes", "oui"):
        return True

    if text in ("false", "0", "no", "non"):
        return False

    return None


def close_enough(a, b, tolerance=TOLERANCE):
    a = as_number(a)
    b = as_number(b)

    return (
        pd.notna(a)
        and pd.notna(b)
        and np.isclose(a, b, atol=tolerance, rtol=0)
    )


def risk_status(risk_pct, observations):
    risk_pct = as_number(risk_pct)
    observations = as_number(observations)

    if (
        pd.isna(risk_pct)
        or pd.isna(observations)
        or observations < MIN_VALID_OBSERVATIONS
    ):
        return "donnees_insuffisantes"

    if risk_pct <= RISK_THRESHOLD_PCT + 1e-9:
        return "admissible"

    return "non_admissible"


def validate_plans(best, pricing, risk):

    required_best = {
        "vm_id",
        "selected_method",
        "selected_plan",
        "current_plan",
        "current_cost",
        "optimized_cost",
        "monthly_savings",
        "validation_risk_pct",
        "test_risk_pct",
        "admissible_validation",
        "admissible_test",
    }

    required_risk = {
        "vm_id",
        "split",
        "selected_plan",
        "plan_risk_pct",
        "valid_observations",
    }

    required_pricing = {
        "vm_id",
        "plan",
        "plan_cpu",
        "plan_ram_gb",
        "monthly_cost",
    }

    for name, df, required in [
        ("best", best, required_best),
        ("risk", risk, required_risk),
        ("pricing", pricing, required_pricing),
    ]:
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} : colonnes manquantes : "
                f"{sorted(missing)}"
            )

    if best["vm_id"].duplicated().any():
        raise ValueError(
            "VM dupliquees dans les recommandations."
        )

    if risk.duplicated(["vm_id", "split"]).any():
        raise ValueError(
            "VM/split dupliques dans plan_risk."
        )

    if pricing.duplicated(["vm_id", "plan"]).any():
        raise ValueError(
            "VM/plan dupliques dans pricing."
        )

    risk_index = risk.set_index(["vm_id", "split"])
    pricing_index = pricing.set_index(["vm_id", "plan"])

    results = []

    for _, row in best.iterrows():

        vm_id = str(row["vm_id"])
        method = str(row["selected_method"]).lower()
        plan = row["selected_plan"]
        current_plan = row["current_plan"]

        current_cost = as_number(row["current_cost"])
        optimized_cost = as_number(row["optimized_cost"])
        savings = as_number(row["monthly_savings"])

        reasons = []

        # ----------------------------------------------
        # 1. Verifier le forfait et ses capacites
        # ----------------------------------------------

        plan_exists = (
            pd.notna(plan)
            and str(plan) in CATALOG
        )

        if plan_exists:
            plan = str(plan)
            plan_cpu, plan_ram, catalog_cost = CATALOG[plan]
        else:
            plan_cpu = np.nan
            plan_ram = np.nan
            catalog_cost = np.nan
            reasons.append("Forfait introuvable")

        current_exists = (
            pd.notna(current_plan)
            and str(current_plan) in CATALOG
        )

        if current_exists:
            current_catalog_cost = CATALOG[
                str(current_plan)
            ][2]
        else:
            current_catalog_cost = np.nan
            reasons.append("Forfait actuel introuvable")

        # ----------------------------------------------
        # 2. Verifier les prix
        # ----------------------------------------------

        price_ok = close_enough(
            optimized_cost, catalog_cost
        )

        current_price_ok = close_enough(
            current_cost, current_catalog_cost
        )

        savings_ok = close_enough(
            savings, current_cost - optimized_cost
        )

        cost_ok = (
            pd.notna(current_cost)
            and pd.notna(optimized_cost)
            and optimized_cost <= current_cost + TOLERANCE
        )

        if not price_ok:
            reasons.append("Prix optimise incoherent")

        if not current_price_ok:
            reasons.append("Prix initial incoherent")

        if not savings_ok:
            reasons.append("Economies incoherentes")

        if not cost_ok:
            reasons.append("Cout optimise superieur")

        # ----------------------------------------------
        # 3. Verifier le catalogue des candidats
        # ----------------------------------------------

        candidate_exists = (
            plan_exists
            and (vm_id, plan) in pricing_index.index
        )

        if candidate_exists:
            candidate = pricing_index.loc[
                (vm_id, plan)
            ]

            candidate_ok = (
                close_enough(
                    candidate["plan_cpu"], plan_cpu
                )
                and close_enough(
                    candidate["plan_ram_gb"], plan_ram
                )
                and close_enough(
                    candidate["monthly_cost"], catalog_cost
                )
            )

            if not candidate_ok:
                reasons.append(
                    "Candidat catalogue incoherent"
                )

        else:
            candidate_ok = False
            reasons.append(
                "Forfait absent des candidats"
            )

        # ----------------------------------------------
        # 4. Controler les risques des deux splits
        # ----------------------------------------------

        split_results = {}

        for split in ("validation", "test"):

            key = (vm_id, split)

            if key not in risk_index.index:

                split_results[split] = {
                    "risk": np.nan,
                    "observations": 0,
                    "status": "donnees_absentes",
                    "consistent": False,
                    "admissible": False,
                }

                reasons.append(
                    f"Risque {split} absent"
                )

                continue

            measured = risk_index.loc[key]

            measured_plan = measured["selected_plan"]
            measured_risk = as_number(
                measured["plan_risk_pct"]
            )
            observations = as_number(
                measured["valid_observations"]
            )

            reported_risk = as_number(
                row[f"{split}_risk_pct"]
            )

            reported_admissible = as_bool(
                row[f"admissible_{split}"]
            )

            status = risk_status(
                measured_risk,
                observations
            )

            calculated_admissible = (
                status == "admissible"
            )

            consistent = (
                str(measured_plan) == str(plan)
                and (
                    close_enough(
                        measured_risk,
                        reported_risk
                    )
                    or (
                        pd.isna(measured_risk)
                        and pd.isna(reported_risk)
                    )
                )
                and reported_admissible
                    == calculated_admissible
            )

            if not consistent:
                reasons.append(
                    f"Incoherence risque {split}"
                )

            split_results[split] = {
                "risk": measured_risk,
                "observations": observations,
                "status": status,
                "consistent": consistent,
                "admissible": calculated_admissible,
            }

        val = split_results["validation"]
        test = split_results["test"]

        # ----------------------------------------------
        # 5. Selection et generalisation
        # ----------------------------------------------

        selection_consistent = (
            plan_exists
            and current_exists
            and price_ok
            and current_price_ok
            and savings_ok
            and cost_ok
            and candidate_ok
            and val["consistent"]
            and test["consistent"]
        )

        # Une selection peut etre coherente tout en
        # n'etant pas admissible au seuil de 1 %.
        #
        # Le test final est descriptif : il ne sert
        # pas a changer la selection du forfait.

        generalization_failure = (
            val["admissible"]
            and not test["admissible"]
        )

        if generalization_failure:
            reasons.append(
                "Admissible validation, echec test"
            )

        if method == "current" and plan != str(current_plan):
            reasons.append(
                "Methode current : forfait different"
            )
            selection_consistent = False

        results.append({
            "vm_id": vm_id,
            "selected_method": method,
            "selected_plan": plan,
            "current_plan": current_plan,

            "plan_cpu": plan_cpu,
            "plan_ram_gb": plan_ram,

            "current_cost": current_cost,
            "optimized_cost": optimized_cost,
            "monthly_savings": savings,

            "price_ok": bool(price_ok),
            "current_price_ok": bool(current_price_ok),
            "savings_ok": bool(savings_ok),
            "cost_ok": bool(cost_ok),
            "candidate_ok": bool(candidate_ok),

            "validation_risk_pct": val["risk"],
            "validation_observations": val["observations"],
            "validation_status": val["status"],
            "admissible_validation": val["admissible"],

            "test_risk_pct": test["risk"],
            "test_observations": test["observations"],
            "test_status": test["status"],
            "admissible_test": test["admissible"],

            "validation_consistent": val["consistent"],
            "test_consistent": test["consistent"],
            "selection_consistent": selection_consistent,
            "generalization_failure": generalization_failure,

            # 'valid' signifie ici coherence des
            # fichiers, pas garantie de faible risque.
            "valid": selection_consistent,

            "reason": (
                "OK" if not reasons
                else "; ".join(reasons)
            ),
        })

    return pd.DataFrame(results)


def main():

    best = read_csv(BEST_FILE)
    pricing = read_csv(PRICING_FILE)
    risk = read_csv(RISK_FILE)

    result = validate_plans(
        best, pricing, risk
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("CONTROLE DES FORFAITS - CLOUDCOST V3")
    print("=" * 60)

    print("VM controlees :", len(result))

    print(
        "Selections coherentes :",
        int(result["selection_consistent"].sum())
    )

    print(
        "Selections incoherentes :",
        int((~result["selection_consistent"]).sum())
    )

    print(
        "Admissibles validation :",
        int(result["admissible_validation"].sum())
    )

    print(
        "Admissibles test final :",
        int(result["admissible_test"].sum())
    )

    print(
        "Echecs de generalisation :",
        int(result["generalization_failure"].sum())
    )

    print("\n===== VM 1, 10, 17, 312 =====")

    selected = result[
        result["vm_id"].isin(
            ["1", "10", "17", "312"]
        )
    ]

    columns = [
        "vm_id",
        "selected_plan",
        "validation_risk_pct",
        "test_risk_pct",
        "admissible_validation",
        "admissible_test",
        "selection_consistent",
        "reason",
    ]

    print(
        selected[columns]
        .round(3)
        .to_string(index=False)
    )

    print("\n===== ANOMALIES DE COHERENCE =====")

    anomalies = result[
        ~result["selection_consistent"]
    ]

    if anomalies.empty:
        print("Aucune incoherence detectee.")
    else:
        print(
            anomalies[
                ["vm_id", "selected_plan", "reason"]
            ]
            .head(20)
            .to_string(index=False)
        )

    print("\nFichier cree :", OUTPUT_FILE)


if __name__ == "__main__":
    main()
