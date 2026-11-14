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
Project/                      <- git repo root (branch: main)
├── .gitignore
├── PROJECT_SPEC.md           <- this file
├── Project_Proposal.pdf      <- source of truth
├── data/                     <- tracked; the project does not run without it
├── logs/                     <- generated, git-ignored
└── util/                     <- all scripts live here
    ├── 00_preprocessing.py   <- library module (nothing executes on import)
    └── 01_data_report.py     <- runnable; writes logs/01_data_report.log
```

### Conventions (team rules, not negotiable)
- **All scripts go in `util/`**, numbered `00` … `0n` in dependency order.
- **No config file and no `argparse`** — constants live at the top of the module.
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
| `util/00_preprocessing.py` | paths, constants, loaders, `CorrelationFilter`, `build_preprocessor`, `build_model_pipeline` | done |
| `util/01_data_report.py` | data exploration report | done, runs clean |
| `util/02_*` | scaffold grouping | not started — blocked on RDKit |

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

Run `util/01_data_report.py` to regenerate all of this.

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
  (acyclic drugs). Empty scaffolds are therefore a real case, not a hypothetical: see § 7.

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
  `.venv/bin/python Project/util/01_data_report.py`.
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

## 7. Open items [OPEN]

- **Correlation filter threshold.** Implemented as `CorrelationFilter`, off by default
  (`correlation_threshold=None`). Intended to be grid-searched via
  `preprocess__drop_correlated__threshold: [0.90, 0.95, 0.99]`. Note that "no filter at all"
  cannot be a grid value, since `None` removes the step entirely — compare it as a separate variant.
- **Empty Murcko scaffolds** (acyclic molecules): the proposal says handling is "predefined" but does
  not say what it is. Suggested default — give each such molecule its own group, keyed by its SMILES,
  so acyclic drugs are not merged into one large group.
- **Hyperparameter grids** for the five algorithms are not specified anywhere; agree on them before
  the tuning run.
- **Tracked logs.** `logs/` is git-ignored to avoid merge conflicts across five people. If the team
  wants the report shared, commit one curated copy at submission with `git add -f`.
