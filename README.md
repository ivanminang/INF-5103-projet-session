
# Cloud Cost Analyzer
### Projet de session — INF-5103 | Concepts statistiques et science des données

## 1. Présentation du projet

**Cloud Cost Analyzer** est un projet d'analyse statistique et d'aide à la décision visant à étudier l'utilisation des ressources informatiques dans des environnements virtualisés et cloud.

Le projet exploite des données réelles de machines virtuelles issues du jeu de données **Bitbrains**, notamment des mesures d'utilisation du processeur (CPU).

L'objectif est d'identifier les possibilités d'optimisation des ressources, d'évaluer les risques associés aux changements de capacité et d'estimer les économies potentielles.

Une application interactive développée avec **Python et Streamlit** permet d'explorer les résultats des analyses.

## 2. Objectifs

Les principaux objectifs du projet sont :

- Analyser statistiquement l'utilisation des ressources des machines virtuelles.
- Étudier les distributions et les variations de consommation CPU.
- Identifier les machines virtuelles potentiellement sous-utilisées ou surutilisées.
- Examiner des scénarios d'optimisation des ressources.
- Évaluer les risques associés aux recommandations.
- Estimer les économies potentielles selon différentes hypothèses.
- Présenter les résultats dans une interface interactive.

## 3. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Développement et analyse des données |
| Pandas | Manipulation des données |
| NumPy | Calcul numérique |
| Matplotlib | Visualisation statistique |
| Streamlit | Application interactive |
| Jupyter Notebook | Exploration et validation |
| Git et GitHub | Gestion des versions |
| Git LFS | Stockage des fichiers volumineux |
| Parquet | Stockage des données préparées |

Les dépendances Python nécessaires sont précisées dans `requirements.txt`.

## 4. Structure du projet

```text
INF-5103-projet-session/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── bitbrains/
│   │       ├── 1.csv
│   │       ├── 2.csv
│   │       └── ...
│   │
│   └── processed/
│       ├── bitbrains_normalized.parquet
│       ├── bitbrains_train.parquet
│       ├── bitbrains_test.parquet
│       ├── bitbrains_train_v3.parquet
│       ├── bitbrains_test_v3.parquet
│       └── bitbrains_validation_v3.parquet
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_validation.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   └── cloudcost/
│       ├── audit_v3.py
│       ├── backtest.py
│       ├── bootstrap_savings.py
│       ├── ingest.py
│       ├── mcnemar_test.py
│       ├── plan_candidates.py
│       ├── plan_risk.py
│       ├── plan_validation.py
│       ├── pricing.py
│       ├── recommend.py
│       ├── sensitivity_analysis.py
│       ├── split_data.py
│       ├── visualization.py
│       └── vm_statistics.py
│
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

## 5. Prérequis

Avant de lancer le projet, installer :

- Python 3.11 ou une version compatible avec les dépendances du projet.
- Git.
- Git LFS.
- Un navigateur Web.
- Un éditeur de code, par exemple Visual Studio Code (facultatif).

Prévoir environ 2 Go d'espace disque disponible pour les données et une capacité supplémentaire pour l'environnement Python et les traitements.

## 6. Installation du projet

### Étape 1 — Télécharger le dépôt GitHub

Ouvrir un terminal et exécuter :

```bash
git lfs install
git clone https://github.com/ivanminang/INF-5103-projet-session.git
```

Accéder ensuite au dossier :

```bash
cd INF-5103-projet-session
```

### Étape 2 — Télécharger les données volumineuses

Les fichiers CSV et Parquet sont gérés avec Git LFS.

Exécuter :

```bash
git lfs pull
```

Cette commande permet de récupérer les fichiers de données associés au dépôt.

**Important :** les données doivent être entièrement téléchargées avant de lancer les analyses.

### Étape 3 — Créer un environnement virtuel Python

```bash
python -m venv .venv
```

Activer l'environnement.

**Windows — Git Bash :**

```bash
source .venv/Scripts/activate
```

**Windows — PowerShell :**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS :**

```bash
source .venv/bin/activate
```

### Étape 4 — Installer les dépendances

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Attendre la fin de l'installation avant de continuer.

## 7. Lancement de l'application

Depuis le dossier principal du projet, avec l'environnement virtuel activé, exécuter :

```bash
python -m streamlit run app/streamlit_app.py
```

L'application devrait démarrer et s'ouvrir automatiquement dans le navigateur.

Si le navigateur ne s'ouvre pas automatiquement, accéder à :

**http://localhost:8501**

L'utilisateur pourra ensuite explorer les fonctionnalités disponibles dans l'interface.

### Arrêter l'application

Dans le terminal où Streamlit fonctionne, appuyer sur :

```text
Ctrl + C
```

## 8. Données utilisées

### 8.1 Données brutes

Les données originales sont situées dans :

```text
data/raw/bitbrains/
```

Ce répertoire contient les fichiers CSV des machines virtuelles du jeu de données Bitbrains.

### 8.2 Données préparées

Les données transformées sont situées dans :

```text
data/processed/
```

Elles comprennent notamment :

- `bitbrains_normalized.parquet`
- `bitbrains_train.parquet`
- `bitbrains_test.parquet`
- `bitbrains_train_v3.parquet`
- `bitbrains_test_v3.parquet`
- `bitbrains_validation_v3.parquet`

Ces fichiers permettent de conserver les données préparées et les partitions utilisées pour les analyses et les validations.

### 8.3 Git LFS

Les fichiers volumineux sont stockés à l'aide de Git Large File Storage.

Les extensions suivies comprennent :

```text
*.csv
*.parquet
```

Pour vérifier les fichiers gérés par Git LFS :

```bash
git lfs ls-files
```

**Remarque :** le téléchargement ZIP classique depuis GitHub peut ne pas inclure le contenu réel des fichiers LFS. Il est recommandé d'utiliser `git clone` puis `git lfs pull`.

## 9. Organisation des analyses

Le projet comprend plusieurs modules Python organisés dans `src/cloudcost/`.

| Module | Rôle principal |
|---|---|
| `ingest.py` | Ingestion des données |
| `vm_statistics.py` | Statistiques des machines virtuelles |
| `visualization.py` | Visualisations |
| `split_data.py` | Partitionnement des données |
| `recommend.py` | Recommandations |
| `pricing.py` | Analyse des coûts |
| `plan_candidates.py` | Plans candidats |
| `plan_risk.py` | Analyse des risques |
| `plan_validation.py` | Validation des plans |
| `backtest.py` | Évaluation rétrospective |
| `bootstrap_savings.py` | Analyse des économies par bootstrap |
| `sensitivity_analysis.py` | Analyse de sensibilité |
| `mcnemar_test.py` | Test statistique de McNemar |
| `audit_v3.py` | Audit de la version V3 |

## 10. Notebooks et visualisations

Deux notebooks Jupyter sont disponibles :

```text
notebooks/01_exploration.ipynb
notebooks/02_validation.ipynb
```

Ils permettent de consulter les travaux d'exploration et de validation.

Pour ouvrir les notebooks dans Jupyter, utiliser un environnement dans lequel Jupyter est installé.

Les figures générées sont disponibles dans :

```text
reports/figures/
```

Elles comprennent notamment des distributions CPU, des séries temporelles et des comparaisons entre machines virtuelles.

## 11. Vérification de l'installation

Après l'installation, vérifier la présence des principaux fichiers :

```bash
ls app/
ls data/raw/bitbrains/
ls data/processed/
```

Vérifier ensuite l'installation de Streamlit :

```bash
python -m streamlit version
```

Enfin, démarrer l'application :

```bash
python -m streamlit run app/streamlit_app.py
```

## 12. Résolution des problèmes courants

### Problème : `ModuleNotFoundError`

Vérifier que l'environnement virtuel est activé, puis réinstaller les dépendances :

```bash
python -m pip install -r requirements.txt
```

### Problème : données CSV ou Parquet introuvables

Vérifier que les données ont été téléchargées :

```bash
git lfs pull
```

Puis vérifier le contenu des dossiers `data/raw/` et `data/processed/`.

### Problème : `streamlit: command not found`

Utiliser :

```bash
python -m streamlit run app/streamlit_app.py
```

### Problème : le port 8501 est déjà utilisé

Démarrer l'application sur un autre port :

```bash
python -m streamlit run app/streamlit_app.py --server.port 8502
```

Puis ouvrir :

```text
http://localhost:8502
```

### Problème : l'application ne trouve pas les modules Python du projet

Depuis Git Bash, à la racine du dépôt, exécuter :

```bash
export PYTHONPATH="$PWD/src"
python -m streamlit run app/streamlit_app.py
```

Cette configuration permet à Python de rechercher les modules présents dans `src/`.

## 13. Reproductibilité

Afin de faciliter la reproduction des travaux :

- Le code source est disponible sur GitHub.
- Les données brutes et préparées sont associées au dépôt.
- Les dépendances sont documentées dans `requirements.txt`.
- Les notebooks permettent de consulter les analyses.
- Les résultats graphiques sont conservés dans `reports/figures/`.
- L'application Streamlit permet d'explorer les résultats.

La reproduction complète des résultats peut dépendre des versions des bibliothèques, des paramètres d'exécution et des ressources matérielles disponibles.

## 14. Limites du projet

Les résultats obtenus dépendent des caractéristiques des données Bitbrains et des hypothèses utilisées pour l'analyse des ressources et des coûts.

Les recommandations produites doivent être considérées comme une aide à la décision. Leur application à un environnement réel nécessite une validation supplémentaire des contraintes techniques, opérationnelles et financières.

## 15. Auteur et contexte académique

**Projet :** Cloud Cost Analyzer  
**Cours :** INF-5103 — Concepts statistiques  
**Domaine :** Science des données et intelligence artificielle  
**Établissement :** Université du Québec en Outaouais (UQO)  
**Année universitaire :** 2026

## 16. Dépôt GitHub

Code source et données :

https://github.com/ivanminang/INF-5103-projet-session

---

**Cloud Cost Analyzer — Analyse statistique et optimisation des ressources informatiques.**
