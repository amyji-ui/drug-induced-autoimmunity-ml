# Project Spec: Drug-Induced Autoimmunity (DIA) Prediction

**Repository:** `drug-induced-autoimmunity-ml`

Machine-readable spec for AI coding assistants working in this folder.
Source of truth: `Project_Proposal.pdf`.

Tags used below:
- **[PROPOSAL]** — stated in the proposal; do not deviate without asking.
- **[VERIFIED]** — checked directly against the files in `data/`; last confirmed 2026-11-13.
- **[OPEN]** — not specified by the proposal; a default is suggested, confirm before relying on it.

---

## 1. Task summary [PROPOSAL]

- **Problem:** binary classification of drugs as DIA-positive (`1`) or DIA-negative (`0`).
- **Inputs:** RDKit molecular descriptors derived from each drug's SMILES.
- **Goals:**
  1. Predict DIA association from molecular descriptors.
  2. Identify the most informative molecular properties (permutation importance).
- **Context:** DIA is a non-IgE-mediated immune-related adverse drug reaction. It is difficult to study
  because it is idiosyncratic, has diverse clinical manifestations, and lacks specific serological markers.
- **Team:** Silver Fang, Amy Ji, Gwen Zhang, Guanyang Wang, Cindy Mai.

---

## 2. Repository layout and conventions

```
Project/                          <- git repo root (branch: main)
├── .gitignore
├── config.yaml                   <- every setting; edit this, not the code
├── requirements.txt
├── PROJECT_SPEC.md               <- this file
├── Project_Proposal.pdf          <- source of truth
├── data/                         <- tracked; the project does not run without it
├── logs/                         <- generated, git-ignored
├── results/                      <- <model>.csv tracked, pipelines/ git-ignored
└── util/                         <- all scripts live here
    ├── config.py                 <- loads config.yaml
    ├── 00_preprocessing.py       <- pipeline step
    ├── 01_scaffold_grouping.py   <- pipeline step
    ├── 02_tuning.py              <- pipeline step (shared harness)
    ├── data_report.py            <- report
    ├── grouping_report.py        <- report
    └── tuning_report.py          <- report
```

### Conventions (team rules, not negotiable)
- **All scripts go in `util/`**. Pipeline steps are numbered `00` … `0n` in dependency order;
  reports and helpers keep plain names so the numbering reads as the pipeline itself.
- **All settings live in `config.yaml`**, never hard-coded. No `argparse`.
- **At most 3 sentences of comment** per function; prefer a short docstring.
- **Expand code**: multi-line steps with named intermediate variables, no dense one-liners.

### Importing between numbered scripts
Module names cannot begin with a digit, so `import 00_preprocessing` is a `SyntaxError`.
Every script that reuses shared code opens with:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
preprocessing = importlib.import_module("00_preprocessing")
```

### What exists so far
| File | Role | Status |
|---|---|---|
**Only pipeline steps are numbered.** Reports and other side utilities keep plain names, so the
numbering reads as the pipeline itself.

| File | Role | Status |
|---|---|---|
| `config.yaml` | **every setting in the project** (repo root) — edit this, not the code | done |
| `util/config.py` | loader for `config.yaml`; exposes the settings as module-level names | done |
| `util/00_preprocessing.py` | pipeline — loaders, transformers, pipeline builders | done |
| `util/01_scaffold_grouping.py` | pipeline — scaffolds, CV groups, the shared folds | done |
| `util/02_tuning.py` | pipeline — shared grid-search harness; **freeze once agreed** | done, verified on kNN |
| `util/05_xgboost.py` | pipeline — XGBoost grid search, 48 combinations | done, 0.777 ± 0.075 |
| `util/03_*`, `04_*`, `06_*`, `07_*` | pipeline — LR, RF, SVM, kNN | not started |

### Writing a per-model script

Each teammate supplies only an estimator and a grid; folds, scaling, metrics, selection rule
and output format all come from `02_tuning.py` and `config.yaml`.

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

tuning = importlib.import_module("02_tuning")
tuning_report = importlib.import_module("tuning_report")

outcome = tuning.run_grid_search(
    model_name="knn",                      # must match a key under `models:` in config.yaml
    estimator=KNeighborsClassifier(),
    param_grid={"model__n_neighbors": [3, 5, 7, 11, 15],
                "model__metric": ["euclidean", "manhattan"],
                "model__weights": ["uniform", "distance"]},
)
tuning_report.report_grid_search(outcome)
```

Two calls, because computation and reporting are separate: `run_grid_search` fits and persists
and returns a `GridSearchOutcome`, and `report_grid_search` turns that into the log. The harness
contains no logging at all, so a later step can reuse an outcome without writing a report.

Grid keys are pipeline paths: `model__<param>` reaches the classifier. Never add
`preprocess__drop_correlated__threshold` (see § 7).

**Per-fold parameters.** A grid value of `tuning.PER_FOLD` is not searched; it is computed inside
each training fold. Used for XGBoost's `scale_pos_weight`, which the proposal requires to come from
each training fold — a global ratio would be derived partly from held-out labels. The sentinel, not
the resolved number, is stored in `params_json`, so the five folds of one combination still group
together. Per-fold values here range 3.021–3.064 against a global 3.042.

`run_grid_search` is the only function in `02_tuning.py` a model script should call, plus
`load_pipelines(model_name)` for the step 4.6 ensemble. The rest (`score_predictions`,
`expand_param_grid`, `load_training_data`, `evaluate_combination`, `summarize_results`,
`get_metric_columns`, `save_results`, `save_pipelines`) are internals — they are what the
harness fixes so five people cannot diverge, so do not call or override them.

`GridSearchOutcome` fields: `model_name`, `scaled`, `results`, `summary`, `best_params_json`,
`best_pipelines`, `n_combinations`, `n_folds`, `elapsed_seconds`, `results_path`, `pipeline_path`.

Each run writes `results/<model_name>.csv` (one row per combination × fold, all seven metrics),
`results/pipelines/<model_name>.joblib` (the winner's five fitted pipelines — the step 4.6
ensemble), and `logs/tuning_<model_name>.log`.
| `util/data_report.py` | report → `logs/data_report.log` | done, runs clean |
| `util/grouping_report.py` | report → `logs/grouping_report.log` | done, runs clean |
| `util/tuning_report.py` | report → `logs/tuning_<model>.log` | done, verified on kNN |

### Pipeline API

**All settings live in `config.yaml` at the repo root** — no module defines its own constants.
`util/config.py` loads that YAML and exposes the values as module-level names; it has a plain
filename, so it imports normally (`import config`) once `util/` is on `sys.path`, while the
digit-prefixed pipeline modules still need `importlib.import_module`.

To change a setting, edit `config.yaml`. Never hard-code a path, a column name, or the seed.

Names exposed by `config`:
- paths (absolute `Path` objects): `PROJECT_DIR`, `CONFIG_FILE`, `DATA_DIR`, `LOG_DIR`,
  `TRAIN_FILE`, `TEST_FILE`, `DESCRIPTOR_DICTIONARY_FILE`, `RESULTS_DIR`, `PIPELINE_DIR`
- columns: `LABEL_COLUMN`, `SMILES_COLUMN`, `POSITIVE_LABEL`
- reproducibility: `RANDOM_SEED = 42`
- models: `MODELS` (name → `{scale: bool}` for all five), `MODEL_NAMES`, `SCALED_MODELS` (derived)
- preprocessing: `HEAVY_TAILED_COLUMNS = ("Ipc",)`, `CORRELATION_THRESHOLD = None`
- cross-validation: `N_SPLITS = 5`, `ACYCLIC_PREFIX = "acyclic::"`
- evaluation: `DECISION_THRESHOLD = 0.5` (predefined by the proposal, never tuned),
  `SELECTION_METRIC = "roc_auc"` (each algorithm's configuration is chosen on its fold mean)
- reports: `DATA_REPORT_LOG`, `GROUPING_REPORT_LOG`, `REDUNDANCY_THRESHOLDS`, `EXAMPLE_THRESHOLD`,
  `N_REDUNDANCY_EXAMPLES`, `N_OVERLAP_EXAMPLES`

`00_preprocessing.py`
- `load_split(csv_path) -> DataFrame`
- `get_descriptor_columns(dataframe) -> list[str]`
- `split_features_labels(dataframe) -> (features, labels, smiles)`
- `LogTransformColumns(columns=("Ipc",))` — log1p on heavy-tailed descriptors
- `CorrelationFilter(threshold=0.95)` — **not in the model pipeline**; used for the redundancy
  report and to derive importance clusters from `.redundant_pairs_`
- `needs_scaling(model_name) -> bool` — reads `config.yaml`; raises `KeyError` on an unknown name
- `build_preprocessor(scale, correlation_threshold=None) -> Pipeline` — low level, takes a bool
- `build_model_pipeline(estimator, model_name, correlation_threshold=None) -> Pipeline`
  — **pass the model name, not a bool.** Scaling comes from `config.yaml`, so no one can
  accidentally standardize a tree or forget to standardize kNN.
- `configure_logging(log_path)`

`01_scaffold_grouping.py`
- `compute_scaffold(smiles_string) -> str` — `""` for acyclic or unparseable
- `build_scaffold_groups(smiles) -> ndarray` — acyclic drugs get `acyclic::<SMILES>` keys
- `build_cv_folds(features, labels, groups) -> list[(train_idx, val_idx)]`
- `find_scaffold_overlap(train_groups, test_groups) -> list[str]`

Typical use in a tuning script:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

preprocessing = importlib.import_module("00_preprocessing")
grouping = importlib.import_module("01_scaffold_grouping")

train_df = preprocessing.load_split(config.TRAIN_FILE)
X, y, smiles = preprocessing.split_features_labels(train_df)
groups = grouping.build_scaffold_groups(smiles)
folds = grouping.build_cv_folds(X, y, groups)
pipeline = preprocessing.build_model_pipeline(estimator, model_name="rbf_svm")
```

`RANDOM_SEED = 42` lives in `00_preprocessing.py` and is used for both the split and every
stochastic model. Import `build_cv_folds` from `01_scaffold_grouping.py` rather than constructing
a splitter, so all models share identical folds as step 4.4 requires.

---

## 3. Data

### Files [VERIFIED]

| Path | Rows | Columns |
|---|---|---|
| `data/DIA_trainingset_RDKit_descriptors.csv` | 477 | 198 |
| `data/DIA_testset_RDKit_descriptors.csv` | 120 | 198 |
| `data/RDKit_ChemDes.xlsx` | 196 descriptor entries | descriptor dictionary |

- Source cited in the proposal: https://doi.org/10.24432/C5332M
- The proposal states "477 instances, 195 RDKit molecular descriptors". Verified: 477 is the
  **training** set only (597 rows total with test), and there are **196** descriptor columns, not 195.
  Prefer the verified numbers in code; flag the discrepancy in the report.

### Column layout [VERIFIED]

- `Label` — int, the target (`0`/`1`).
- `SMILES` — string identifier of the molecule.
- Remaining 196 columns — numeric RDKit descriptors (`BalabanJ`, `BertzCT`, `Chi0`, …, `fr_urea`).
- Train and test have identical column names in identical order.

### Class balance [VERIFIED]

| Split | Label 0 | Label 1 | Positive rate |
|---|---|---|---|
| Train | 359 | 118 | 24.7% |
| Test | 90 | 30 | 25.0% |

Imbalanced, roughly 3:1. This is why the proposal asks for class-weighted variants and for
ROC AUC / AP / balanced accuracy / MCC rather than plain accuracy.

### Quality checks [VERIFIED]

Run `util/data_report.py` to regenerate all of this.

- No missing or non-finite values in either split.
- No duplicate SMILES within train or within test.
- No SMILES shared between train and test.
- **17 descriptors are constant in train** (`NumRadicalElectrons`, `SMR_VSA8`, `SlogP_VSA9`,
  `VSA_EState1`–`VSA_EState7`, `fr_azide`, `fr_barbitur`, `fr_diazo`, `fr_isocyan`,
  `fr_isothiocyan`, `fr_prisulfonamd`, `fr_thiocyan`). Never hard-code this list —
  `VarianceThreshold` relearns it inside each fold.
- **25 are constant in test, 8 of them test-only** (`fr_N_O`, `fr_SH`, `fr_aldehyde`, `fr_azo`,
  `fr_furan`, `fr_hdrzone`, `fr_oxazole`, `fr_oxime`). These are *kept*, because only the training
  fold decides what is dropped. But each takes one value across all 120 test molecules, so it cannot
  discriminate there — relevant when reading the importance ranking.
- **0 invalid SMILES** in either split — all 597 parse cleanly under RDKit.
- **Scaffolds (train):** 477 molecules → 319 unique Murcko scaffolds, of which **43 are empty**
  (acyclic drugs), giving **361 CV groups** after the acyclic split. Test: 120 molecules,
  106 groups, 10 acyclic.
- **The provided train and test splits share 23 scaffolds.** The test set is therefore *not*
  scaffold-disjoint from training, so the final held-out number in step 4.6 is easier than a
  novel-scaffold screen. Report it as such; the grouped-CV estimate is the conservative one.
- **Measured cost of ignoring scaffolds:** an untuned Random Forest scores ROC AUC 0.833 under
  random `StratifiedKFold` but 0.752 under `StratifiedGroupKFold` — **+0.080 AUC of pure optimism**,
  with fold variance also rising (±0.040 → ±0.079). Illustrative only, but it is why step 4.2 exists.

### Constant vs redundant — distinct concepts

| | constant | redundant |
|---|---|---|
| appearance | one value, zero variance | varies normally |
| information | none at all | real, but duplicated |
| removed by | `VarianceThreshold` | `CorrelationFilter` |
| example | `fr_azide` = 0 for all 477 molecules | `HeavyAtomCount`, \|r\| = 0.996 with `Chi0` |

Order matters: constants are dropped **first**, because a zero-variance column's correlation is
`0/0` = NaN and would pollute the correlation matrix.

### Redundancy [VERIFIED]

Of the 179 descriptors that vary in train:

| threshold | kept | dropped |
|---|---|---|
| \|r\| > 0.99 | 169 | 10 |
| \|r\| > 0.95 | 151 | 28 |
| \|r\| > 0.90 | 142 | 37 |

Redundancy is concentrated in one cluster: **`Chi0` alone displaces 13 descriptors** —
`ExactMolWt`, `HeavyAtomCount`, `HeavyAtomMolWt`, `MolWt`, `LabuteASA`, `Kappa1` and the other
`Chi` variants — all proxies for *molecular size*. `Chi0` wins only because it comes first in
column order, not because it is the best of them. If size ranks high in permutation importance,
report it as "molecular size matters", not as "`Chi0` matters".

### Descriptor dictionary [VERIFIED]

`data/RDKit_ChemDes.xlsx` has one sheet with columns
`Index, Descriptor Name, Type, Description, Dimention, Extended class`, 196 rows.
`Extended class` counts: Constitutional 106, MOE-type 58, Topological 15, Connectivity 12,
Molecular property 5. Use it to attach human-readable names to the top permutation-importance features.
Reading it needs `openpyxl`, which is not installed.

---

## 4. Workflow [PROPOSAL]

### 4.1 Preprocessing
- Check class balance, invalid SMILES, duplicate molecules, non-finite descriptor values.
- Remove constant features; optionally filter highly correlated descriptors.
- Standardize descriptors for Logistic Regression, RBF-SVM, and kNN.
- Fit **all** preprocessing inside each CV training fold and apply to the matching validation fold
  via a pipeline. No fitting on full data before splitting.

### 4.2 Structure grouping
- Compute the Murcko scaffold of each drug from its SMILES, with predefined handling of empty scaffolds.
- Drugs sharing a scaffold stay in the same fold.
- Check group sizes and per-fold class distributions so no single group dominates a fold.
- Check and report scaffold overlap between the provided training and test sets.

### 4.3 Models
1. Logistic Regression — regularized linear baseline.
2. Random Forest — nonlinear interactions, bagged trees.
3. XGBoost — regularized gradient boosting.
4. RBF-SVM — nonlinear decision boundaries.
5. kNN — distance-based, after feature scaling.

- Logistic Regression, Random Forest, SVM: compare unweighted vs class-weighted configurations.
- XGBoost: use `scale_pos_weight` or sample weights computed **from each training fold**.
- kNN: tune neighborhood size, distance metric, distance weighting.
  Distance weighting is *not* class weighting — do not treat it as the imbalance remedy.

### 4.4 Cross-validation and tuning
- 5-fold CV, grouped by scaffold, approximately stratified by label.
- Identical folds reused across every model and parameter combination.
- Score each parameter combination over all five folds; select the highest **mean validation ROC AUC**
  per algorithm.
- Retain the five fitted pipelines of each selected configuration (same hyperparameters, different
  training folds).

### 4.5 Evaluation
- Report ROC AUC, Average Precision, precision, recall, balanced accuracy, MCC, F1.
- Continuous scores for ROC AUC and AP; fixed decision threshold of **0.5** for the classification metrics.
  The threshold is predefined — do not tune it.
- Compare fold-level results, means and standard deviations **descriptively**.
  No Friedman test, no post hoc tests.
- Select the final algorithm from CV results, and state in the report that these are *development*
  estimates, since the same CV guided tuning and selection.

### 4.6 Final ensemble
- The selected algorithm's five fitted pipelines are the final ensemble.
- Average their DIA-positive probabilities per test drug.
  For SVM, do scaffold-grouped probability calibration within each training partition.
- Evaluate the averaged predictions on the test set **once**, with no further tuning.
- Assess informative molecular properties with permutation importance on held-out CV folds,
  and summarize ranking stability across folds.

---

## 5. Rules for an assistant working here

- The test set is touched exactly once, at step 4.6. Never use it for tuning, feature selection,
  threshold choice, or scaler fitting.
- Every transform goes inside a `Pipeline`; nothing is fit outside a CV fold.
- Row-independent work (parsing SMILES, computing a Murcko scaffold) may happen on the full dataset.
  Anything that aggregates across rows (means, variances, correlations, rankings) must be fit in-fold.
- Build the fold assignment once, reuse the same object everywhere, and fix random seeds.
- Scaffold grouping is the unit of splitting; a plain `StratifiedKFold` is incorrect here.
- Keep the metric set complete — do not substitute accuracy.
- Check runtime cost before launching a large grid search; this dataset is small, but a wide grid
  across 5 models × 5 folds adds up. Ask before running anything long or heavily parallel.

---

## 6. Environment

- **Interpreter:** `.venv/` at `09615_code/`, one level **above** the repo root and shared with the
  course labs, so it is not tracked here. Run scripts from `09615_code/` as
  `.venv/bin/python Project/util/data_report.py`.
- **Dependencies:** pinned in `requirements.txt` at the repo root (`uv pip install -r requirements.txt`).
  All of them are installed and verified working, with one exception below.

### Known environment traps

- **`xgboost` needs OpenMP, which pip cannot supply.** On a fresh machine `import xgboost` raises
  `XGBoostError: libxgboost.dylib could not be loaded`; the fix is `brew install libomp`
  (macOS) — not a pip package, so `requirements.txt` alone is not enough to reproduce the
  environment. Already resolved on this machine: `build_info()["USE_OPENMP"]` is `True`.
- **`rdkit` is pinned to 2026.3.6, below the latest release.** In 2026.09.2,
  `Chem.MurckoDecompose` raises `RuntimeError: bad any cast`, which breaks
  `MurckoScaffold.MurckoScaffoldSmiles` and therefore all of step 4.2. The rest of RDKit is
  unaffected, so the failure is easy to misread. Do not upgrade past 2026.3.6 without re-testing
  scaffold computation. Python 3.14 wheels exist only for 2026.3.x and 2026.9.x, so there is no
  older fallback.
- **scikit-learn 1.9 removed `BaseEstimator._validate_data`**; use
  `sklearn.utils.validation.validate_data(self, X, ...)` in custom transformers.

---

## 7. Decisions made

- **Empty Murcko scaffolds → one group each** (2026-11-14). The 43 acyclic training drugs get group
  keys `acyclic::<SMILES>`. Rationale: grouping exists to stop *structural analogs* straddling a
  split, and two acyclic drugs are not analogs — they share only the absence of rings. Keeping `''`
  as a single group would make it the largest group in the dataset (43, vs benzene's 36) and fill
  45% of one fold.
- **Redundant features → kept; no correlation filter** (2026-11-14). The proposal calls the filter
  optional, so this is compliant. `build_preprocessor` keeps `correlation_threshold=None` as its
  default, so `CorrelationFilter` is never added to the model pipeline; it stays in the codebase
  only to power the redundancy section of `data_report.py`. Do not add
  `preprocess__drop_correlated__threshold` to any grid. Two consequences are now binding, see § 8.
- **`Ipc` → `log1p` before scaling** (2026-11-14). Implemented as `LogTransformColumns`, first step
  of the pipeline. Raw `Ipc` spans 3.2–8.5e13 (variance 1.8e25, twenty orders of magnitude above any
  other descriptor) and accounted for **100%** of unscaled kNN/SVM distance; z-scoring alone still
  left one molecule 19.8 SD out. After the transform the worst single-feature share of distance is
  **0.56%** and the extreme molecule sits at 3.8 SD. Row-independent, so fold-safe anywhere.

## 8. Open items [OPEN]

- **Permutation importance MUST be computed at cluster level** — this is now required, not advisory,
  because the redundant descriptors are staying. Permuting `MolWt` alone barely moves the score,
  since `Chi0` and 12 other size proxies still carry the signal, so all 14 score near zero even if
  molecular size is the dominant driver. Permute each correlated cluster together. Clusters can be
  derived from `CorrelationFilter(threshold=0.95).fit(...).redundant_pairs_`, which already records
  which descriptor displaced which.
- **kNN and RBF-SVM are handicapped by the kept redundancy**, and the report must say so. Molecular
  size is represented ~14 times, so it carries ~14x weight in every Euclidean distance. If those two
  models underperform the trees, that is partly an artifact of feature duplication, not evidence that
  distance-based methods suit this problem poorly.
- **Hyperparameter grids** for the five algorithms are not specified anywhere; agree on them before
  the tuning run.
- **Tracked logs.** `logs/` is git-ignored to avoid merge conflicts across five people. If the team
  wants the report shared, commit one curated copy at submission with `git add -f`.
