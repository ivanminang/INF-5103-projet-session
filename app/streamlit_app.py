
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# CLOUDCOST ANALYZER V3
# TABLEAU DE BORD STREAMLIT
# Projet INF-5103 - UQO
# ============================================================

st.set_page_config(
    page_title="CloudCost Analyzer V3 | INF-5103",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# DESIGN SYSTEM : présentation uniquement, aucun calcul modifié
# ------------------------------------------------------------
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#1765D1", "#14A89B", "#F59E0B",
    "#7B61C9", "#F07167", "#64748B"
]
px.defaults.height = 355

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --brand:#1765D1;
  --ink:#18253D;
  --muted:#63738B;
}

html, body, [class*="css"], [data-testid="stApp"] {
  font-family:'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
  background:#F5F8FD;
}

[data-testid="stMainBlockContainer"] {
  max-width:1440px;
  padding-top:1.5rem;
  padding-bottom:3rem;
}

[data-testid="stSidebar"] {
  background:#FFFFFF;
  border-right:1px solid #E5ECF5;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color:#53637C;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
  border-radius:11px;
  padding:0.48rem 0.55rem;
  margin:0.09rem 0;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background:#EAF2FF;
  color:#1755AF;
  font-weight:700;
}

h1 {
  color:var(--ink);
  font-weight:800 !important;
  letter-spacing:-0.045em;
  font-size:2rem !important;
}

h2 {
  color:var(--ink);
  font-weight:750 !important;
  letter-spacing:-0.025em;
  font-size:1.35rem !important;
}

h3 {
  color:#263955;
  font-weight:700 !important;
  font-size:1.08rem !important;
}

[data-testid="stMetric"] {
  background:#FFFFFF;
  border:1px solid #E5ECF5;
  border-radius:15px;
  padding:1.05rem 1.15rem;
  min-height:112px;
  box-shadow:0 3px 14px rgba(24,53,91,.035);
}

[data-testid="stMetricLabel"] {
  color:#66768D;
  font-weight:600;
}

[data-testid="stMetricValue"] {
  color:#172A48;
  font-weight:800;
  letter-spacing:-0.04em;
}

[data-testid="stMetricDelta"] {
  font-weight:650;
}

[data-testid="stPlotlyChart"] {
  background:#FFFFFF;
  border:1px solid #E5ECF5;
  border-radius:16px;
  padding:0.5rem;
}

[data-testid="stDataFrame"] {
  border:1px solid #E5ECF5;
  border-radius:12px;
  overflow:hidden;
}

[data-testid="stAlert"] {
  border-radius:11px;
}

[data-testid="stTabs"] button {
  font-weight:650;
}

[data-testid="stDownloadButton"] button {
  border-radius:10px;
  font-weight:650;
}

[data-testid="stExpander"] {
  border-radius:12px;
  background:#FFFFFF;
}

hr {
  border-color:#E5ECF5 !important;
}

.hero {
  background:linear-gradient(
    112deg,
    #123B78 0%,
    #1765D1 64%,
    #2389E3 100%
  );
  color:#FFFFFF;
  padding:1.4rem 1.7rem;
  border-radius:19px;
  margin-bottom:1.25rem;
  box-shadow:0 9px 24px rgba(23,101,209,.12);
}

.hero-title {
  font-size:1.6rem;
  font-weight:800;
  letter-spacing:-.035em;
  margin:0;
}

.hero-subtitle {
  font-size:.89rem;
  color:#DDEAFF;
  margin:.38rem 0 0;
}

.hero-tag {
  display:inline-block;
  background:rgba(255,255,255,.16);
  border:1px solid rgba(255,255,255,.2);
  border-radius:20px;
  padding:.2rem .7rem;
  font-size:.72rem;
  font-weight:700;
  margin-bottom:.55rem;
}

.section-note {
  color:#63738B;
  font-size:.88rem;
  margin:-.25rem 0 1rem;
}

@media (max-width:700px) {
  .hero {
    padding:1rem;
  }

  .hero-title {
    font-size:1.25rem;
  }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. CONFIGURATION
# ============================================================

HERE = Path(__file__).resolve().parent

BASE_DIR = (
    HERE
    if (HERE / "data" / "processed").exists()
    else HERE.parent
)

DATA_DIR = BASE_DIR / "data" / "processed"

RISK_THRESHOLD_PCT = 1.0
MIN_OBSERVATIONS = 100

FILES = {
    "metrics": "bitbrains_normalized.parquet",
    "stats": "vm_statistics.csv",
    "recommendations": "vm_recommendations_v3.csv",
    "validation": "vm_backtest_validation_v3.csv",
    "test": "vm_backtest_test_v3.csv",
    "pricing": "vm_pricing_v3.csv",
    "best": "vm_best_recommendations_v3.csv",
    "plan_validation": "vm_plan_validation_v3.csv",
    "plan_risk": "vm_plan_risk_v3.csv",
    "mcnemar": "mcnemar_results_v3.csv",
    "mcnemar_pairs": "mcnemar_pairs_v3.csv",
    "bootstrap": "bootstrap_results_v3.csv",
    "bootstrap_samples": "bootstrap_samples_v3.csv",
    "sensitivity": "sensitivity_results_v3.csv",
    "sensitivity_details": "sensitivity_details_v3.csv",
    "audit": "audit_coherence_v3.csv",
}


# ============================================================
# 2. CHARGEMENT DES DONNEES
# ============================================================

@st.cache_data(show_spinner=False)
def load_file(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return None

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(
            path,
            dtype={"vm_id": str},
        )

    if "vm_id" in df.columns:
        df["vm_id"] = df["vm_id"].astype(str)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

    return df


data = {
    key: load_file(filename)
    for key, filename in FILES.items()
}

metrics = data["metrics"]
stats = data["stats"]
recommendations = data["recommendations"]
validation = data["validation"]
test = data["test"]
pricing = data["pricing"]
best = data["best"]
plan_validation = data["plan_validation"]
plan_risk = data["plan_risk"]


# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================

def money(value):
    if pd.isna(value):
        return "N/D"
    return f"{float(value):,.2f} $"


def pct(value):
    if pd.isna(value):
        return "N/D"
    return f"{float(value):.2f} %"


def show(df):
    if df is None:
        st.info("Données indisponibles.")
        return

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
    )


def get_vm_rows(df, vm_id):
    if (
        df is None
        or vm_id is None
        or "vm_id" not in df.columns
    ):
        return pd.DataFrame()

    return df[
        df["vm_id"].astype(str) == str(vm_id)
    ].copy()


def download(df, filename):
    if df is None:
        return

    st.download_button(
        label=f"Télécharger {filename}",
        data=df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}",
    )


def bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "oui"])
    )


# ============================================================
# 3. FONCTIONS UTILITAIRES (SUITE)
# ============================================================

def risk_summary(df):
    if df is None or df.empty:
        return pd.DataFrame()

    required = {"method", "global_risk", "vm_id"}

    if not required.issubset(df.columns):
        return pd.DataFrame()

    tmp = df.copy()

    tmp["global_risk"] = pd.to_numeric(
        tmp["global_risk"],
        errors="coerce",
    )

    if "valid_observations" in tmp.columns:
        eligible = (
            pd.to_numeric(
                tmp["valid_observations"],
                errors="coerce",
            ) >= MIN_OBSERVATIONS
        )
    else:
        eligible = pd.Series(
            False,
            index=tmp.index,
        )

    tmp["admissible_calcule"] = (
        eligible
        & tmp["global_risk"].notna()
        & (
            tmp["global_risk"]
            <= RISK_THRESHOLD_PCT / 100
        )
    )

    summary = (
        tmp.groupby("method")
        .agg(
            risque_moyen=("global_risk", "mean"),
            configurations=("vm_id", "count"),
            vm_admissibles=("admissible_calcule", "sum"),
        )
        .reset_index()
    )

    summary["risque_moyen_pct"] = (
        summary["risque_moyen"] * 100
    )

    return summary


def add_risk_line(fig):
    fig.add_hline(
        y=RISK_THRESHOLD_PCT,
        line_dash="dash",
        annotation_text="Seuil 1 %",
    )
    return fig


def selected_status(df, column):
    if df is None or column not in df.columns:
        return 0

    return int(bool_series(df[column]).sum())


# ============================================================
# 4. NAVIGATION
# ============================================================

st.sidebar.markdown("### ☁️ CloudCost Analyzer")
st.sidebar.caption("**V3 · Laboratoire de décision cloud**")

NAV_ICONS = {
    "Vue globale": "📊  Vue globale",
    "Analyse des VM": "🖥️  Analyse des VM",
    "Recommandations A/B": "🧪  Recommandations A/B",
    "Optimisation économique": "💰  Optimisation économique",
    "Validation des forfaits": "🛡️  Validation des forfaits",
    "Analyse statistique": "📈  Analyse statistique",
    "Audit et généralisation": "🔎  Audit et généralisation",
}

page = st.sidebar.radio(
    "Navigation",
    [
        "Vue globale",
        "Analyse des VM",
        "Recommandations A/B",
        "Optimisation économique",
        "Validation des forfaits",
        "Analyse statistique",
        "Audit et généralisation",
    ],
    format_func=lambda value: NAV_ICONS.get(value, value),
)

vm_ids = []

source_ids = (
    best
    if best is not None and "vm_id" in best.columns
    else metrics
)

if source_ids is not None and "vm_id" in source_ids.columns:
    vm_ids = sorted(
        source_ids["vm_id"]
        .dropna()
        .astype(str)
        .unique(),
        key=lambda x: (
            (0, int(x))
            if x.isdigit()
            else (1, x)
        ),
    )

selected_vm = (
    st.sidebar.selectbox(
        "Machine virtuelle",
        vm_ids,
    )
    if vm_ids
    else None
)

st.sidebar.divider()

st.sidebar.caption(
    "Dataset : Bitbrains (2013)\n\n"
    "Catalogue fictif (USD/mois)\n\n"
    "Seuil de risque : 1 %\n\n"
    "Minimum : 100 observations valides\n\n"
    "Projet INF-5103 — UQO"
)

# ------------------------------------------------------------
# Bandeau commun à toutes les pages
# ------------------------------------------------------------

PAGE_SUBTITLES = {
    "Vue globale":
        "Synthèse des coûts, des ressources et de l'admissibilité",

    "Analyse des VM":
        "Exploration temporelle de l'utilisation CPU et mémoire",

    "Recommandations A/B":
        "Comparaison expérimentale des stratégies de dimensionnement",

    "Optimisation économique":
        "Catalogue fictif, économies et choix des forfaits",

    "Validation des forfaits":
        "Contrôle des capacités et du risque sur validation et test",

    "Analyse statistique":
        "McNemar, bootstrap et robustesse des décisions",

    "Audit et généralisation":
        "Traçabilité des sélections et comportement hors échantillon",
}

st.markdown(
    '<div class="hero">'
    '<span class="hero-tag">INF-5103 · UQO · VERSION 3</span>'
    '<div class="hero-title">☁️ CloudCost Analyzer</div>'
    '<div class="hero-subtitle">'
    + PAGE_SUBTITLES.get(page, "")
    + '</div></div>',
    unsafe_allow_html=True,
)


# ============================================================
# 5. VUE GLOBALE
# ============================================================

if page == "Vue globale":

    st.title("☁️ CloudCost Analyzer V3")

    st.subheader(
        "Optimisation statistique des ressources Cloud"
    )

    if best is None or best.empty:
        st.warning(
            "Fichier vm_best_recommendations_v3.csv absent."
        )
        st.stop()

    current_cost = pd.to_numeric(
        best["current_cost"],
        errors="coerce",
    ).sum()

    optimized_cost = pd.to_numeric(
        best["optimized_cost"],
        errors="coerce",
    ).sum()

    savings = current_cost - optimized_cost

    savings_pct = (
        savings / current_cost * 100
        if current_cost > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Indicateurs économiques
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "VM analysées",
        best["vm_id"].nunique(),
    )

    c2.metric(
        "Coût initial / mois",
        money(current_cost),
    )

    c3.metric(
        "Coût optimisé / mois",
        money(optimized_cost),
    )

    c4.metric(
        "Économies simulées",
        money(savings),
        pct(savings_pct),
    )

    st.info(
        "Les tarifs sont fictifs. Les économies présentées "
        "sont des estimations expérimentales et non des "
        "économies réellement réalisées."
    )

    # --------------------------------------------------------
    # Admissibilité
    # --------------------------------------------------------

    st.subheader(
        "Admissibilité des forfaits sélectionnés"
    )

    a, b, c = st.columns(3)

    a.metric(
        "VM analysées",
        len(best),
    )

    b.metric(
        "Admissibles validation",
        selected_status(
            best,
            "admissible_validation",
        ),
    )

    c.metric(
        "Admissibles test final",
        selected_status(
            best,
            "admissible_test",
        ),
    )

    st.divider()

    # --------------------------------------------------------
    # Comparaison des coûts
    # --------------------------------------------------------

    st.subheader("Comparaison des coûts")

    costs = pd.DataFrame({
        "Situation": [
            "Avant optimisation",
            "Après optimisation",
        ],
        "Coût mensuel": [
            current_cost,
            optimized_cost,
        ],
    })

    fig = px.bar(
        costs,
        x="Situation",
        y="Coût mensuel",
        color="Situation",
        text="Coût mensuel",
        title="Coûts mensuels simulés",
        color_discrete_map={
            "Avant optimisation": "#64748B",
            "Après optimisation": "#14A89B",
        },
    )

    fig.update_traces(
        texttemplate="%{text:,.0f} $",
        textposition="outside",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Répartition des décisions
    # --------------------------------------------------------

    st.subheader("Décisions sélectionnées")

    counts = (
        best["selected_method"]
        .value_counts()
        .rename_axis("Décision")
        .reset_index(name="Nombre de VM")
    )

    fig = px.pie(
        counts,
        names="Décision",
        values="Nombre de VM",
        hole=0.45,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Comparaison des risques
    # --------------------------------------------------------

    st.subheader("Comparaison des risques A/B")

    for title, df in [
        ("Validation", validation),
        ("Test final", test),
    ]:

        st.markdown(f"**{title}**")

        show(
            risk_summary(df).round(3)
        )

    st.caption(
        "Le risque mesure les dépassements de capacité "
        "CPU ou RAM, et non les pannes."
    )


# ============================================================
# 6. ANALYSE DES MACHINES VIRTUELLES
# ============================================================

elif page == "Analyse des VM":

    st.title(
        f"Analyse de la VM {selected_vm}"
    )

    if metrics is None or selected_vm is None:
        st.warning(
            "Données VM indisponibles."
        )
        st.stop()

    vm = get_vm_rows(
        metrics,
        selected_vm,
    )

    if vm.empty:
        st.warning(
            "Aucune observation disponible."
        )
        st.stop()

    if "timestamp" in vm.columns:
        vm = vm.sort_values("timestamp")

    # --------------------------------------------------------
    # Statistiques descriptives
    # --------------------------------------------------------

    vm_stats = get_vm_rows(
        stats,
        selected_vm,
    )

    if not vm_stats.empty:

        row = vm_stats.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        for column, label, box in [
            ("cpu_pct_mean", "CPU moyen", c1),
            ("cpu_pct_p95", "CPU P95", c2),
            ("cpu_pct_p99", "CPU P99", c3),
            ("mem_pct_p95", "RAM P95", c4),
        ]:

            box.metric(
                label,
                pct(row[column])
                if column in row.index
                else "N/D",
            )

    st.divider()

    # --------------------------------------------------------
    # Évolution temporelle CPU/RAM
    # --------------------------------------------------------

    for column, title, label in [
        (
            "cpu_pct",
            "Évolution temporelle CPU",
            "CPU (%)",
        ),
        (
            "mem_pct",
            "Évolution temporelle RAM",
            "RAM (%)",
        ),
    ]:

        st.subheader(title)

        if column not in vm.columns:
            st.warning(
                f"Colonne absente : {column}"
            )
            continue

        if "timestamp" in vm.columns:

            fig = px.line(
                vm,
                x="timestamp",
                y=column,
                labels={
                    "timestamp": "Date (UTC)",
                    column: label,
                },
            )

        else:

            fig = px.line(
                vm,
                y=column,
                labels={
                    column: label,
                },
            )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Distributions CPU/RAM
    # --------------------------------------------------------

    for column, title in [
        ("cpu_pct", "Distribution CPU"),
        ("mem_pct", "Distribution RAM"),
    ]:

        if column not in vm.columns:
            continue

        st.subheader(title)

        fig = px.histogram(
            vm.dropna(subset=[column]),
            x=column,
            nbins=50,
            title=f"{title} — VM {selected_vm}",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.caption(
        "Ces graphiques décrivent les observations "
        "disponibles. Les recommandations sont "
        "calculées sur l'ensemble d'apprentissage."
    )


# ============================================================
# 7. RECOMMANDATIONS A/B
# ============================================================

elif page == "Recommandations A/B":

    st.title("🧪 Comparaison des méthodes A et B")

    st.markdown(
        '<p class="section-note">'
        "Comparaison des stratégies de dimensionnement "
        "sur les données de validation et de test final."
        "</p>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Présentation des méthodes
    # --------------------------------------------------------

    col_a, col_b = st.columns(2)

    with col_a:
        st.info(
            "**Méthode A — Heuristique CPU**\n\n"
            "Réduction théorique de moitié des cœurs CPU "
            "lorsque l'utilisation moyenne est inférieure "
            "à 20 %. La RAM reste inchangée."
        )

    with col_b:
        st.info(
            "**Méthode B — Percentile P95**\n\n"
            "Dimensionnement selon les percentiles P95 "
            "CPU et RAM, avec une marge de sécurité "
            "de 15 %."
        )

    vm_rec = get_vm_rows(
        recommendations,
        selected_vm,
    )

    vm_validation = get_vm_rows(
        validation,
        selected_vm,
    )

    vm_test = get_vm_rows(
        test,
        selected_vm,
    )

    if vm_rec.empty:
        st.warning(
            "Recommandations indisponibles "
            "pour cette machine virtuelle."
        )
        st.stop()

    rec = vm_rec.iloc[0]

    # --------------------------------------------------------
    # Comparaison des configurations
    # --------------------------------------------------------

    st.subheader(
        f"Configurations proposées — VM {selected_vm}"
    )

    comparison = pd.DataFrame([
        {
            "Méthode": "Configuration actuelle",
            "CPU (cœurs)": rec.get(
                "current_cpu_cores",
                np.nan,
            ),
            "RAM (GiB)": rec.get(
                "current_ram_gb",
                np.nan,
            ),
        },
        {
            "Méthode": "Méthode A",
            "CPU (cœurs)": rec.get(
                "a_cpu_cores",
                np.nan,
            ),
            "RAM (GiB)": rec.get(
                "a_ram_gb",
                np.nan,
            ),
        },
        {
            "Méthode": "Méthode B",
            "CPU (cœurs)": rec.get(
                "b_cpu_cores",
                np.nan,
            ),
            "RAM (GiB)": rec.get(
                "b_ram_gb",
                np.nan,
            ),
        },
    ])

    show(comparison.round(2))

    # --------------------------------------------------------
    # Résultats de validation et test
    # --------------------------------------------------------

    tab_validation, tab_test = st.tabs([
        "📘 Validation",
        "🧪 Test final",
    ])

    for tab, title, df in [
        (
            tab_validation,
            "Résultats sur validation",
            vm_validation,
        ),
        (
            tab_test,
            "Résultats sur test final",
            vm_test,
        ),
    ]:

        with tab:

            st.subheader(title)

            if df.empty:
                st.info(
                    "Aucun résultat disponible."
                )
                continue

            df = df.copy()

            if "global_risk" in df.columns:
                df["Risque (%)"] = (
                    pd.to_numeric(
                        df["global_risk"],
                        errors="coerce",
                    ) * 100
                )

            if "valid_observations" in df.columns:
                df["Observations suffisantes"] = (
                    pd.to_numeric(
                        df["valid_observations"],
                        errors="coerce",
                    ) >= MIN_OBSERVATIONS
                )

            columns = [
                "method",
                "cpu_cores",
                "ram_gb",
                "Risque (%)",
                "admissible",
                "Observations suffisantes",
            ]

            columns = [
                c for c in columns
                if c in df.columns
            ]

            show(
                df[columns].round(3)
            )

    # --------------------------------------------------------
    # Graphique comparatif des risques
    # --------------------------------------------------------

    risk_frames = []

    for label, df in [
        ("Validation", vm_validation),
        ("Test final", vm_test),
    ]:

        if (
            not df.empty
            and "global_risk" in df.columns
        ):
            risk_frames.append(
                df.assign(
                    Ensemble=label
                )
            )

    if risk_frames:

        risk_plot = pd.concat(
            risk_frames,
            ignore_index=True,
        )

        risk_plot["Risque (%)"] = (
            pd.to_numeric(
                risk_plot["global_risk"],
                errors="coerce",
            ) * 100
        )

        st.subheader(
            "Comparaison des risques CPU/RAM"
        )

        fig = px.bar(
            risk_plot,
            x="method",
            y="Risque (%)",
            color="Ensemble",
            barmode="group",
            title=(
                "Risque de dépassement "
                f"— VM {selected_vm}"
            ),
            labels={
                "method": "Méthode",
            },
        )

        fig = add_risk_line(fig)

        fig.update_layout(
            height=380,
            legend_title_text="",
            margin=dict(
                l=15,
                r=15,
                t=55,
                b=15,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.caption(
        "Le risque correspond à la proportion "
        "d'observations présentant un dépassement "
        "de capacité CPU ou RAM. "
        "Il ne représente pas un taux de panne."
    )


# ============================================================
# 8. OPTIMISATION ECONOMIQUE
# ============================================================

elif page == "Optimisation économique":

    st.title("💰 Optimisation économique")

    st.markdown(
        '<p class="section-note">'
        "Comparaison des coûts et identification "
        "des forfaits commerciaux simulés."
        "</p>",
        unsafe_allow_html=True,
    )

    st.warning(
        "Les tarifs sont fictifs et utilisés "
        "uniquement à des fins pédagogiques."
    )

    if best is None or pricing is None:
        st.warning(
            "Données économiques indisponibles."
        )
        st.stop()

    # --------------------------------------------------------
    # Indicateurs économiques
    # --------------------------------------------------------

    initial_cost = pd.to_numeric(
        best["current_cost"],
        errors="coerce",
    ).sum()

    final_cost = pd.to_numeric(
        best["optimized_cost"],
        errors="coerce",
    ).sum()

    total_savings = initial_cost - final_cost

    savings_rate = (
        total_savings / initial_cost * 100
        if initial_cost > 0
        else np.nan
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Coût initial",
        money(initial_cost),
    )

    c2.metric(
        "Coût optimisé",
        money(final_cost),
    )

    c3.metric(
        "Économies mensuelles",
        money(total_savings),
    )

    c4.metric(
        "Réduction des coûts",
        pct(savings_rate),
    )

    st.divider()

    # --------------------------------------------------------
    # Classement des économies
    # --------------------------------------------------------

    st.subheader(
        "Classement des économies par VM"
    )

    ranked = best.sort_values(
        "monthly_savings",
        ascending=False,
    ).copy()

    columns = [
        "vm_id",
        "selected_method",
        "selected_plan",
        "current_cost",
        "optimized_cost",
        "monthly_savings",
        "savings_pct",
        "validation_risk_pct",
        "test_risk_pct",
        "admissible_test",
    ]

    available_columns = [
        c for c in columns
        if c in ranked.columns
    ]

    top_n = st.slider(
        "Nombre de VM à afficher",
        min_value=10,
        max_value=min(100, len(ranked)),
        value=min(30, len(ranked)),
        step=10,
    )

    top_vms = ranked.head(top_n)

    show(
        top_vms[
            available_columns
        ].round(2)
    )

    # --------------------------------------------------------
    # Graphique des économies
    # --------------------------------------------------------

    st.subheader(
        f"Top {top_n} des économies mensuelles"
    )

    fig = px.bar(
        top_vms,
        x="vm_id",
        y="monthly_savings",
        color="monthly_savings",
        color_continuous_scale=[
            "#BBD8FA",
            "#1765D1",
        ],
        labels={
            "vm_id": "Machine virtuelle",
            "monthly_savings": "USD/mois",
        },
        title="Économies mensuelles simulées",
    )

    fig.update_layout(
        height=380,
        coloraxis_showscale=False,
        xaxis_type="category",
        margin=dict(
            l=15,
            r=15,
            t=55,
            b=15,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------------
    # Forfaits candidats de la VM
    # --------------------------------------------------------

    st.subheader(
        f"Forfaits candidats — VM {selected_vm}"
    )

    vm_price = get_vm_rows(
        pricing,
        selected_vm,
    )

    if vm_price.empty:

        st.info(
            "Aucun forfait candidat disponible "
            "pour cette VM."
        )

    else:

        candidate_columns = [
            "plan",
            "plan_cpu",
            "plan_ram_gb",
            "monthly_cost",
            "plan_risk_pct",
            "valid_observations",
            "admissible",
        ]

        available = [
            c for c in candidate_columns
            if c in vm_price.columns
        ]

        display = vm_price[available].copy()

        if "monthly_cost" in display.columns:
            display = display.sort_values(
                "monthly_cost"
            )

        show(
            display.round(3)
        )

    # --------------------------------------------------------
    # Forfait sélectionné
    # --------------------------------------------------------

    st.subheader(
        f"Forfait retenu — VM {selected_vm}"
    )

    vm_best = get_vm_rows(
        best,
        selected_vm,
    )

    if not vm_best.empty:

        row = vm_best.iloc[0]

        a, b, c = st.columns(3)

        a.metric(
            "Forfait sélectionné",
            str(
                row.get(
                    "selected_plan",
                    "N/D",
                )
            ),
        )

        b.metric(
            "Coût optimisé",
            money(
                row.get(
                    "optimized_cost",
                    np.nan,
                )
            ),
        )

        c.metric(
            "Économie mensuelle",
            money(
                row.get(
                    "monthly_savings",
                    np.nan,
                )
            ),
        )

        with st.expander(
            "Voir toutes les informations "
            "de la recommandation"
        ):
            show(
                vm_best.round(3)
            )

    st.divider()

    download(
        best,
        FILES["best"],
    )


# ============================================================
# 9. VALIDATION DES FORFAITS
# ============================================================

elif page == "Validation des forfaits":

    st.title("🛡️ Validation des forfaits V3")

    st.markdown(
        '<p class="section-note">'
        "Évaluation des configurations sélectionnées "
        "sur les ensembles de validation et de test final."
        "</p>",
        unsafe_allow_html=True,
    )

    if best is None:
        st.warning(
            "Fichier des forfaits retenus absent."
        )
        st.stop()

    total = len(best)

    valid_val = selected_status(
        best,
        "admissible_validation",
    )

    valid_test = selected_status(
        best,
        "admissible_test",
    )

    # --------------------------------------------------------
    # Indicateurs de validation
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "VM évaluées",
        total,
    )

    c2.metric(
        "Admissibles validation",
        valid_val,
        pct(
            valid_val / total * 100
            if total else np.nan
        ),
    )

    c3.metric(
        "Admissibles test final",
        valid_test,
        pct(
            valid_test / total * 100
            if total else np.nan
        ),
    )

    # --------------------------------------------------------
    # Comparaison graphique
    # --------------------------------------------------------

    st.subheader(
        "Admissibilité des forfaits sélectionnés"
    )

    admissibility_chart = pd.DataFrame({
        "Ensemble": [
            "Validation",
            "Test final",
        ],
        "Admissibles": [
            valid_val,
            valid_test,
        ],
        "Non admissibles ou données insuffisantes": [
            total - valid_val,
            total - valid_test,
        ],
    })

    fig = px.bar(
        admissibility_chart,
        x="Ensemble",
        y=[
            "Admissibles",
            "Non admissibles ou données insuffisantes",
        ],
        barmode="stack",
        title="Admissibilité sur validation et test",
        labels={
            "value": "Nombre de VM",
            "variable": "Statut",
        },
        color_discrete_sequence=[
            "#14A89B",
            "#F59E0B",
        ],
    )

    fig.update_layout(
        height=370,
        legend_title_text="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Tableau des forfaits
    # --------------------------------------------------------

    st.subheader(
        "Forfaits sélectionnés"
    )

    display_columns = [
        "vm_id",
        "selected_method",
        "selected_plan",
        "validation_risk_pct",
        "test_risk_pct",
        "admissible_validation",
        "admissible_test",
    ]

    available_columns = [
        c for c in display_columns
        if c in best.columns
    ]

    filter_status = st.selectbox(
        "Filtrer les résultats",
        [
            "Toutes les VM",
            "Admissibles sur test",
            "Non admissibles sur test",
        ],
    )

    filtered_best = best.copy()

    if filter_status == "Admissibles sur test":

        filtered_best = filtered_best[
            bool_series(
                filtered_best["admissible_test"]
            )
        ]

    elif filter_status == "Non admissibles sur test":

        filtered_best = filtered_best[
            ~bool_series(
                filtered_best["admissible_test"]
            )
        ]

    show(
        filtered_best[
            available_columns
        ].round(3)
    )

    st.divider()

    # --------------------------------------------------------
    # Analyse individuelle
    # --------------------------------------------------------

    st.subheader(
        f"Analyse détaillée — VM {selected_vm}"
    )

    vm_best = get_vm_rows(
        best,
        selected_vm,
    )

    if not vm_best.empty:

        row = vm_best.iloc[0]

        a, b, c = st.columns(3)

        a.metric(
            "Forfait sélectionné",
            str(
                row.get(
                    "selected_plan",
                    "N/D",
                )
            ),
        )

        b.metric(
            "Risque validation",
            pct(
                row.get(
                    "validation_risk_pct",
                    np.nan,
                )
            ),
        )

        c.metric(
            "Risque test",
            pct(
                row.get(
                    "test_risk_pct",
                    np.nan,
                )
            ),
        )

        risks = pd.DataFrame({
            "Ensemble": [
                "Validation",
                "Test final",
            ],
            "Risque (%)": [
                row.get(
                    "validation_risk_pct",
                    np.nan,
                ),
                row.get(
                    "test_risk_pct",
                    np.nan,
                ),
            ],
        })

        fig = px.bar(
            risks,
            x="Ensemble",
            y="Risque (%)",
            color="Ensemble",
            title=(
                "Risque du forfait retenu "
                f"— VM {selected_vm}"
            ),
            color_discrete_sequence=[
                "#1765D1",
                "#14A89B",
            ],
        )

        fig = add_risk_line(fig)

        fig.update_layout(
            height=350,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Contrôles complémentaires
    # --------------------------------------------------------

    st.subheader(
        "Contrôle complémentaire des forfaits"
    )

    if plan_validation is not None:

        with st.expander(
            "Afficher les résultats détaillés "
            "des contrôles"
        ):

            show(
                plan_validation.round(3)
            )

            download(
                plan_validation,
                FILES["plan_validation"],
            )

    else:

        st.info(
            "Contrôle complémentaire indisponible."
        )

    # --------------------------------------------------------
    # Risques CPU/RAM détaillés
    # --------------------------------------------------------

    st.subheader(
        "Risques détaillés CPU et RAM"
    )

    vm_risk = get_vm_rows(
        plan_risk,
        selected_vm,
    )

    if not vm_risk.empty:

        show(
            vm_risk.round(3)
        )

    else:

        st.info(
            "Aucune mesure détaillée de risque "
            "disponible pour cette VM."
        )

    st.divider()

    download(
        best,
        FILES["best"],
    )

    st.caption(
        "L'admissibilité correspond au respect "
        "des critères expérimentaux sur les données "
        "évaluées. Elle ne garantit pas la disponibilité "
        "en production."
    )


# ============================================================
# 10. ANALYSE STATISTIQUE
# ============================================================

elif page == "Analyse statistique":

    st.title("📈 Analyse statistique")

    st.markdown(
        '<p class="section-note">'
        "Évaluation de la significativité, de "
        "l'incertitude et de la robustesse des résultats."
        "</p>",
        unsafe_allow_html=True,
    )

    st.info(
        "Trois analyses complémentaires : test exact "
        "de McNemar, bootstrap des économies et analyse "
        "de sensibilité. Les résultats proviennent "
        "des fichiers CSV calculés en amont."
    )

    tab1, tab2, tab3 = st.tabs([
        "🧪 Test de McNemar",
        "📊 Bootstrap",
        "🎛️ Analyse de sensibilité",
    ])

    # --------------------------------------------------------
    # 10.1 TEST EXACT DE MCNEMAR
    # --------------------------------------------------------

    with tab1:

        st.header("Test exact de McNemar")

        st.write(
            "Comparaison appariée de l'admissibilité "
            "des méthodes A et B sur les mêmes VM "
            "de l'ensemble de test final."
        )

        result = data["mcnemar"]
        pairs = data["mcnemar_pairs"]

        if result is None or result.empty:

            st.warning(
                "Fichier mcnemar_results_v3.csv absent "
                "ou vide."
            )

        else:

            row = result.iloc[0]

            a, b, c = st.columns(3)

            a.metric(
                "VM comparées",
                int(row["vm_compared"]),
            )

            b.metric(
                "Admissibilité A",
                pct(row["a_admissible_pct"]),
            )

            c.metric(
                "Admissibilité B",
                pct(row["b_admissible_pct"]),
            )

            st.divider()

            st.subheader("Tableau de contingence")

            table = pd.DataFrame(
                [
                    [
                        row["both_admissible"],
                        row["a_only_admissible"],
                    ],
                    [
                        row["b_only_admissible"],
                        row["neither_admissible"],
                    ],
                ],
                index=[
                    "A admissible",
                    "A non admissible",
                ],
                columns=[
                    "B admissible",
                    "B non admissible",
                ],
            )

            show(
                table.reset_index(
                    names="Méthode A"
                )
            )

            st.subheader("Résultat du test")

            p_value = float(row["p_value"])

            c1, c2 = st.columns(2)

            c1.metric(
                "Valeur p",
                f"{p_value:.3e}",
            )

            c2.metric(
                "Seuil de significativité",
                "5 %",
            )

            if p_value < 0.05:

                st.success(
                    "La différence d'admissibilité "
                    "entre les méthodes A et B est "
                    "statistiquement significative "
                    "au seuil de 5 %."
                )

            else:

                st.info(
                    "La différence d'admissibilité "
                    "n'est pas statistiquement "
                    "significative au seuil de 5 %."
                )

            # Interprétation automatique
            a_rate = float(row["a_admissible_pct"])
            b_rate = float(row["b_admissible_pct"])

            if p_value < 0.05:

                if a_rate > b_rate:

                    st.markdown(
                        "**Interprétation :** la méthode A "
                        "présente une proportion "
                        "d'admissibilité significativement "
                        "plus élevée que la méthode B "
                        "sur les VM comparées."
                    )

                elif b_rate > a_rate:

                    st.markdown(
                        "**Interprétation :** la méthode B "
                        "présente une proportion "
                        "d'admissibilité significativement "
                        "plus élevée que la méthode A "
                        "sur les VM comparées."
                    )

            st.caption(
                "Le test exact de McNemar compare "
                "les résultats binaires appariés. "
                "Il ne compare pas directement "
                "les économies financières."
            )

            download(
                result,
                FILES["mcnemar"],
            )

        if pairs is not None:

            with st.expander(
                "Afficher les résultats appariés par VM"
            ):

                show(pairs)

                download(
                    pairs,
                    FILES["mcnemar_pairs"],
                )

    # --------------------------------------------------------
    # 10.2 BOOTSTRAP DES ECONOMIES
    # --------------------------------------------------------

    with tab2:

        st.header("Bootstrap des économies")

        st.write(
            "Estimation de l'incertitude des économies "
            "simulées par rééchantillonnage des VM."
        )

        result = data["bootstrap"]
        samples = data["bootstrap_samples"]

        if result is None or result.empty:

            st.warning(
                "Fichier bootstrap_results_v3.csv absent "
                "ou vide."
            )

        else:

            row = result.iloc[0]

            a, b, c = st.columns(3)

            a.metric(
                "VM analysées",
                int(row["vm_count"]),
            )

            b.metric(
                "Économies mensuelles",
                money(row["monthly_savings"]),
            )

            c.metric(
                "Réduction des coûts",
                pct(row["savings_pct"]),
            )

            st.divider()

            st.subheader(
                "Intervalle de confiance bootstrap à 95 %"
            )

            ci = pd.DataFrame([
                {
                    "Indicateur":
                        "Économies totales USD/mois",
                    "Estimation":
                        row["monthly_savings"],
                    "Borne inférieure":
                        row["monthly_savings_ci_low"],
                    "Borne supérieure":
                        row["monthly_savings_ci_high"],
                },
                {
                    "Indicateur":
                        "Réduction des coûts (%)",
                    "Estimation":
                        row["savings_pct"],
                    "Borne inférieure":
                        row["savings_pct_ci_low"],
                    "Borne supérieure":
                        row["savings_pct_ci_high"],
                },
                {
                    "Indicateur":
                        "Économie moyenne par VM",
                    "Estimation":
                        row["mean_savings_per_vm"],
                    "Borne inférieure":
                        row["mean_savings_ci_low"],
                    "Borne supérieure":
                        row["mean_savings_ci_high"],
                },
            ])

            show(ci.round(2))

            # Interprétation automatique
            low = float(row["monthly_savings_ci_low"])
            high = float(row["monthly_savings_ci_high"])

            st.info(
                "**Interprétation :** selon la procédure "
                "bootstrap utilisée, l'intervalle de "
                "confiance à 95 % des économies mensuelles "
                f"simulées s'étend de {money(low)} "
                f"à {money(high)}."
            )

            download(
                result,
                FILES["bootstrap"],
            )

        # ----------------------------------------------------
        # Distribution des réplications
        # ----------------------------------------------------

        if samples is not None and not samples.empty:

            st.subheader(
                "Distribution des réplications bootstrap"
            )

            options = {
                "Économies totales":
                    "monthly_savings",
                "Réduction (%)":
                    "savings_pct",
                "Économie moyenne par VM":
                    "mean_savings_per_vm",
            }

            available_options = {
                label: column
                for label, column in options.items()
                if column in samples.columns
            }

            if available_options:

                selected = st.selectbox(
                    "Indicateur à visualiser",
                    list(available_options.keys()),
                )

                column = available_options[selected]

                values = pd.to_numeric(
                    samples[column],
                    errors="coerce",
                ).dropna()

                if not values.empty:

                    fig = px.histogram(
                        x=values,
                        nbins=50,
                        title=(
                            "Distribution bootstrap — "
                            + selected
                        ),
                        labels={
                            "x": selected,
                            "y": "Fréquence",
                        },
                        color_discrete_sequence=[
                            "#1765D1"
                        ],
                    )

                    low = values.quantile(0.025)
                    high = values.quantile(0.975)

                    fig.add_vline(
                        x=low,
                        line_dash="dash",
                        line_color="#F59E0B",
                        annotation_text="2,5 %",
                    )

                    fig.add_vline(
                        x=high,
                        line_dash="dash",
                        line_color="#14A89B",
                        annotation_text="97,5 %",
                    )

                    fig.update_layout(
                        height=390,
                        showlegend=False,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

            download(
                samples,
                FILES["bootstrap_samples"],
            )

        st.caption(
            "Le bootstrap rééchantillonne les VM, "
            "et non les observations temporelles. "
            "L'interprétation suppose que les VM "
            "sont suffisamment représentatives "
            "et indépendantes."
        )

    # --------------------------------------------------------
    # 10.3 ANALYSE DE SENSIBILITE
    # --------------------------------------------------------

    with tab3:

        st.header("Analyse de sensibilité")

        st.write(
            "Influence du seuil de risque et de la marge "
            "de sécurité sur les économies simulées."
        )

        result = data["sensitivity"]
        details = data["sensitivity_details"]

        if result is None or result.empty:

            st.warning(
                "Fichier sensitivity_results_v3.csv "
                "absent ou vide."
            )

        else:

            sens = result.copy()

            margins = sorted(
                sens["margin_pct"].dropna().unique()
            )

            selected_margins = st.multiselect(
                "Marges de sécurité (%)",
                margins,
                default=list(margins),
            )

            filtered = sens[
                sens["margin_pct"].isin(
                    selected_margins
                )
            ].copy()

            if not filtered.empty:

                filtered["Marge"] = (
                    filtered["margin_pct"]
                    .map(lambda x: f"{x:g} %")
                )

                st.subheader(
                    "Économies selon le seuil de risque"
                )

                fig = px.line(
                    filtered.sort_values(
                        "threshold_pct"
                    ),
                    x="threshold_pct",
                    y="monthly_savings",
                    color="Marge",
                    markers=True,
                    labels={
                        "threshold_pct":
                            "Seuil de risque (%)",
                        "monthly_savings":
                            "Économies USD/mois",
                    },
                    title=(
                        "Sensibilité des économies "
                        "aux paramètres"
                    ),
                )

                fig.update_layout(
                    height=400,
                    legend_title_text="Marge de sécurité",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                with st.expander(
                    "Afficher le tableau des scénarios"
                ):

                    show(
                        filtered.drop(
                            columns="Marge"
                        ).round(2)
                    )

            # ------------------------------------------------
            # Scénario de référence
            # ------------------------------------------------

            reference = sens.loc[
                np.isclose(
                    sens["margin_pct"],
                    15,
                )
                & np.isclose(
                    sens["threshold_pct"],
                    1,
                )
            ]

            if not reference.empty:

                r = reference.iloc[0]

                st.subheader(
                    "Scénario de référence"
                )

                st.caption(
                    "Marge de sécurité : 15 % | "
                    "Seuil de risque : 1 %"
                )

                a, b, c = st.columns(3)

                a.metric(
                    "Économies simulées",
                    money(r["monthly_savings"]),
                )

                b.metric(
                    "Réduction des coûts",
                    pct(r["savings_pct"]),
                )

                c.metric(
                    "VM optimisées",
                    int(r["optimized_vm"]),
                )

            # ------------------------------------------------
            # Interprétation
            # ------------------------------------------------

            if "monthly_savings" in sens.columns:

                minimum = pd.to_numeric(
                    sens["monthly_savings"],
                    errors="coerce",
                ).min()

                maximum = pd.to_numeric(
                    sens["monthly_savings"],
                    errors="coerce",
                ).max()

                st.info(
                    "**Interprétation :** les économies "
                    "mensuelles simulées varient "
                    f"de {money(minimum)} "
                    f"à {money(maximum)} "
                    "dans les scénarios étudiés."
                )

            download(
                result,
                FILES["sensitivity"],
            )

        if details is not None:

            with st.expander(
                "Afficher les détails des scénarios "
                "par machine virtuelle"
            ):

                show(details)

                download(
                    details,
                    FILES["sensitivity_details"],
                )

        st.caption(
            "Les scénarios de sensibilité imposent "
            "les besoins P95 de la méthode B. "
            "Ils sont distincts de l'optimisation "
            "libre du catalogue."
        )


# ============================================================
# 11. AUDIT ET GENERALISATION
# ============================================================

elif page == "Audit et généralisation":

    st.title("🔎 Audit et généralisation")

    st.markdown(
        '<p class="section-note">'
        "Vérification de la cohérence des décisions "
        "et de leur comportement sur les données "
        "de test final."
        "</p>",
        unsafe_allow_html=True,
    )

    if best is None or best.empty:

        st.error(
            "Fichier vm_best_recommendations_v3.csv "
            "absent ou vide."
        )

        st.stop()

    # --------------------------------------------------------
    # 11.1 COHERENCE DES SELECTIONS
    # --------------------------------------------------------

    st.subheader("1. Cohérence des sélections")

    total = len(best)

    unique_vms = best["vm_id"].nunique()

    duplicates = int(
        best["vm_id"].duplicated().sum()
    )

    a, b, c = st.columns(3)

    a.metric(
        "VM analysées",
        total,
    )

    b.metric(
        "VM uniques",
        unique_vms,
    )

    c.metric(
        "Doublons",
        duplicates,
    )

    if duplicates == 0:

        st.success(
            "Aucun doublon de VM "
            "dans les sélections."
        )

    else:

        st.error(
            f"{duplicates} doublon(s) détecté(s)."
        )

    # --------------------------------------------------------
    # 11.2 COHERENCE ECONOMIQUE
    # --------------------------------------------------------

    st.subheader("2. Cohérence économique")

    current_cost = pd.to_numeric(
        best["current_cost"],
        errors="coerce",
    ).sum()

    optimized_cost = pd.to_numeric(
        best["optimized_cost"],
        errors="coerce",
    ).sum()

    savings = current_cost - optimized_cost

    a, b, c = st.columns(3)

    a.metric(
        "Coût initial",
        money(current_cost),
    )

    b.metric(
        "Coût optimisé",
        money(optimized_cost),
    )

    c.metric(
        "Économies simulées",
        money(savings),
    )

    # --------------------------------------------------------
    # 11.3 ADMISSIBILITE
    # --------------------------------------------------------

    st.subheader(
        "3. Admissibilité validation/test"
    )

    val_count = selected_status(
        best,
        "admissible_validation",
    )

    test_count = selected_status(
        best,
        "admissible_test",
    )

    a, b, c = st.columns(3)

    a.metric(
        "VM évaluées",
        total,
    )

    b.metric(
        "Admissibles validation",
        val_count,
    )

    c.metric(
        "Admissibles test",
        test_count,
    )

    summary = pd.DataFrame({
        "Ensemble": [
            "Validation",
            "Test final",
        ],
        "VM admissibles": [
            val_count,
            test_count,
        ],
        "VM non admissibles ou insuffisantes": [
            total - val_count,
            total - test_count,
        ],
    })

    fig = px.bar(
        summary,
        x="Ensemble",
        y=[
            "VM admissibles",
            "VM non admissibles ou insuffisantes",
        ],
        barmode="stack",
        title="Admissibilité des forfaits",
        color_discrete_sequence=[
            "#14A89B",
            "#F59E0B",
        ],
    )

    fig.update_layout(
        height=370,
        legend_title_text="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 11.4 ECHECS DE GENERALISATION
    # --------------------------------------------------------

    st.subheader("4. Échecs de généralisation")

    val_ok = bool_series(
        best["admissible_validation"]
    )

    test_ok = bool_series(
        best["admissible_test"]
    )

    failure_mask = val_ok & ~test_ok

    failures = best.loc[
        failure_mask
    ].copy()

    failure_count = len(failures)

    retained_count = int(
        (val_ok & test_ok).sum()
    )

    initially_valid = int(
        val_ok.sum()
    )

    failure_rate = (
        failure_count / initially_valid * 100
        if initially_valid > 0
        else np.nan
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Admissibles validation",
        initially_valid,
    )

    b.metric(
        "Restées admissibles sur test",
        retained_count,
    )

    c.metric(
        "Échecs de généralisation",
        failure_count,
    )

    d.metric(
        "Taux d'échec",
        pct(failure_rate),
    )

    st.info(
        "Un échec de généralisation signifie qu'un "
        "forfait admissible sur validation ne l'est "
        "plus sur le test final. Il ne s'agit pas "
        "d'une panne observée."
    )

    columns = [
        "vm_id",
        "selected_plan",
        "selected_method",
        "validation_risk_pct",
        "test_risk_pct",
        "admissible_validation",
        "admissible_test",
    ]

    if failures.empty:

        st.success(
            "Aucun échec de généralisation détecté."
        )

    else:

        st.subheader(
            "VM présentant un échec de généralisation"
        )

        show(
            failures[
                [
                    c for c in columns
                    if c in failures.columns
                ]
            ].round(3)
        )

        download(
            failures,
            "generalization_failures_v3.csv",
        )

    # --------------------------------------------------------
    # 11.5 RISQUES PAR VM
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        f"5. Analyse détaillée — VM {selected_vm}"
    )

    vm_best = get_vm_rows(
        best,
        selected_vm,
    )

    vm_risk = get_vm_rows(
        plan_risk,
        selected_vm,
    )

    vm_control = get_vm_rows(
        plan_validation,
        selected_vm,
    )

    if not vm_best.empty:

        row = vm_best.iloc[0]

        a, b, c = st.columns(3)

        a.metric(
            "Forfait sélectionné",
            str(
                row.get(
                    "selected_plan",
                    "N/D",
                )
            ),
        )

        b.metric(
            "Risque validation",
            pct(
                row.get(
                    "validation_risk_pct",
                    np.nan,
                )
            ),
        )

        c.metric(
            "Risque test",
            pct(
                row.get(
                    "test_risk_pct",
                    np.nan,
                )
            ),
        )

        risks = pd.DataFrame({
            "Ensemble": [
                "Validation",
                "Test final",
            ],
            "Risque (%)": [
                row.get(
                    "validation_risk_pct",
                    np.nan,
                ),
                row.get(
                    "test_risk_pct",
                    np.nan,
                ),
            ],
        })

        fig = px.bar(
            risks,
            x="Ensemble",
            y="Risque (%)",
            color="Ensemble",
            title=(
                "Risque du forfait retenu "
                f"— VM {selected_vm}"
            ),
            color_discrete_sequence=[
                "#1765D1",
                "#14A89B",
            ],
        )

        fig = add_risk_line(fig)

        fig.update_layout(
            height=350,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    if not vm_risk.empty:

        st.markdown(
            "**Risques CPU et RAM détaillés**"
        )

        show(
            vm_risk.round(3)
        )

        if "status" in vm_risk.columns:

            st.caption(
                "Statuts possibles : admissible, "
                "non_admissible, donnees_insuffisantes "
                "ou forfait_introuvable."
            )

    if not vm_control.empty:

        with st.expander(
            "Contrôle de cohérence de cette VM"
        ):

            show(
                vm_control.round(3)
            )

    # --------------------------------------------------------
    # 11.6 RAPPORT D'AUDIT GLOBAL
    # --------------------------------------------------------

    st.divider()

    st.subheader("6. Rapport d'audit global")

    audit = data["audit"]

    if audit is None:

        st.warning(
            "Rapport audit_coherence_v3.csv absent."
        )

        st.code(
            "python -m src.cloudcost.audit_v3",
            language="bash",
        )

    elif audit.empty:

        st.success(
            "Le rapport d'audit ne contient "
            "aucune anomalie."
        )

        download(
            audit,
            FILES["audit"],
        )

    else:

        st.warning(
            f"{len(audit)} signalement(s) "
            "dans le rapport d'audit."
        )

        if "controle" in audit.columns:

            counts = (
                audit["controle"]
                .value_counts()
                .rename_axis("Contrôle")
                .reset_index(
                    name="Signalements"
                )
            )

            show(counts)

            fig = px.bar(
                counts,
                x="Contrôle",
                y="Signalements",
                title=(
                    "Répartition des signalements "
                    "d'audit"
                ),
            )

            fig.update_layout(
                height=350,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        show(audit)

        download(
            audit,
            FILES["audit"],
        )

    st.caption(
        "L'audit de cohérence, l'admissibilité "
        "statistique et la généralisation sont "
        "trois contrôles distincts. Un audit sans "
        "incohérence ne signifie pas que tous "
        "les forfaits respectent le seuil de risque."
    )


# ============================================================
# 12. PIED DE PAGE
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "☁️ CloudCost Analyzer V3\n\n"
    "Projet INF-5103 — UQO\n\n"
    "Les risques représentent des dépassements "
    "de capacité, et non des pannes."
)

st.markdown("---")

st.caption(
    "CloudCost Analyzer V3 | "
    "Projet universitaire INF-5103 — UQO | "
    "Données Bitbrains (2013) | "
    "Tarification fictive | "
    "Résultats expérimentaux"
)
