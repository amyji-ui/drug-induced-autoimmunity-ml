# Primary goals

Configure this checkout of drug-induced-autoimmunity-ml for local cluster use.

Publish the existing Logistic Regression implementation on branch `RL`, compatible with the latest kNN/XGBoost integration.

## TODO

- [x] Clone the requested repository into `DRUG_CL/drug_cl/`.
- [x] Read project conventions and verify the compute-node allocation.
- [x] Create an isolated Python environment and install dependencies.
- [x] Verify imports, training-data preprocessing, scaffold folds, and a small model fit.
- [x] Document reproducible setup and run commands.
- [x] Implement the approved Logistic Regression grid and persisted shared CV folds.
- [x] Run 70 fits on the allocated compute node and check convergence.
- [x] Verify saved models/metrics and report all seven metrics with class-weight comparison.
- [x] Review remote kNN/XGBoost branches and base `RL` on current main.
- [x] Revalidate Logistic Regression and compatibility on the current Slurm allocation.
- [x] Push `RL` and verify its remote commit.

## Milestone results and reproduction notes

- 2026-11-14 (America/New_York): Cloned https://github.com/amyji-ui/drug-induced-autoimmunity-ml.git at commit `4127846f9d67b96d5fddf2f619410ea6e82cdfe7`.
- Repository includes the descriptor CSVs and dictionary, shared preprocessing/grouping/tuning modules, and an existing kNN results CSV. Individual model entry points are not implemented upstream.
- 2026-11-14: Installed all 24 direct/transitive packages successfully. `uv pip check --python .venv/bin/python` reports all installed packages compatible. Exported exact versions to `requirements-linux.lock.txt`.
- 2026-11-14: Launched training-only environment validation as a Slurm step in job `57864754`, with one CPU and BLAS/OpenMP thread limits of one. Output: `logs/environment_check.log`.
- 2026-11-14: Validation exited successfully (code 0): every dependency and public helper module imports; descriptor dictionary has 196 rows; training data has 477 rows, 196 descriptors, and 361 scaffold groups. All five kNN pipelines completed fit/predict, validation fold sizes were 95/95/96/95/96, and every fold had zero shared groups with its training partition. The held-out test split was not loaded. See `logs/environment_check.log`.
- 2026-11-14: Setup complete. `git diff --exit-code` confirmed existing upstream tracked files are unchanged; additions are setup documentation, progress history, Python/dependency version records, and the environment check helper.
- 2026-11-14: User approved the Logistic Regression plan: L2/lbfgs, C in [0.001, 0.01, 0.1, 1, 10, 100, 1000], class_weight in [None, balanced], max_iter=5000, tol=1e-4, seed=42, and the same five scaffold folds (70 fits). Select by mean validation ROC AUC, report all seven metrics, retain the five winning pipelines, and leave the test split for final algorithm selection.
- 2026-11-14: Implemented the model entry point and YAML grid. Using scikit-learn 1.9's `l1_ratio=0.0` for L2 avoids the deprecated `penalty` parameter. ConvergenceWarning is treated as an error for every fit. Extended the grouping helper to persist/reload folds with a data/order/settings fingerprint and reuse the same list within a process; the shared tuning harness remains unchanged.
- 2026-11-14: Slurm job `57864754` on `n201` completed all 70 LR fits in 4.2 seconds of search time, exit code 0, no convergence warnings. Winner: C=0.01, class_weight=None; five winning fits used 14/15/14/13/13 iterations. Mean ROC AUC=0.6856 (SD 0.0912), AP=0.4417, recall=0.0761 at threshold 0.5. Best balanced model also uses C=0.01: ROC AUC=0.6846 (SD 0.0882), AP=0.4487, recall=0.6257. The predefined ROC AUC rule selects the unweighted model despite the balanced model's better threshold-based metrics. Fold metrics, aggregate summary, shared fold manifest, and winning pipelines were saved; test data were not loaded.
- 2026-11-14: Independent artifact validation passed in the same Slurm allocation: persisted folds load without invoking the splitter and reuse the same in-memory list; indices exactly match the original splitter; changed row order is rejected; all 70 combination/fold rows are unique and complete; all 14 mean/std summaries reproduce; all seven metrics from every reloaded winning pipeline reproduce to 1e-12 absolute tolerance; scaler means and sample counts match each training fold. Validation log: `logs/logistic_regression_validation.log`.
- 2026-11-14: Added `results/logistic_regression_report.md` with both weighting variants and all winning fold scores, and updated `SETUP.md` with the training command and fold-reuse behavior. `git diff --check` passed; the existing shared tuning harness and upstream kNN results are unchanged. No errors occurred during the LR run or artifact validation.

## Environment and path choices

- Project root: `/net/dali/home/chikina/guanyanw/DRUG_CL/drug_cl`.
- Host `n201` is assigned to this user by running Slurm job `57864754` (`big_memory`, 4 CPUs, 60 GB RAM). Installation and validation will run on this compute node.
- Use a project-local `.venv/` instead of the upstream author's machine-specific shared environment. Preserve the scientific configuration in `config.yaml`.
- Validation will use training data only; preserve the upstream tracked `results/knn.csv`.

## Errors encountered and resolutions

- None so far.
- 2026-11-14: An early `tail` of `/tmp/drug-cl-install-guanyanw.log` returned “No such file or directory” because `.venv` creation was still running `ensurepip` and installation had not started. Process inspection confirmed the setup remained active; check for the log before subsequent reads.
- 2026-11-14: `.venv` creation finished and the install log appeared, resolving the early log-read issue. Dependency installation then failed: system Python 3.11.5 is too old for pinned NumPy 2.5.2 (requires Python >=3.12). Install a project-local Python 3.13 using a local uv bootstrap and recreate the unused environment, retaining upstream dependency pins.
- 2026-11-14: An auxiliary PyPI availability check failed while parsing an unrelated historical joblib version (`0.3.2d.dev`). The requested pins checked before this error exist; subsequent checks will query only the requested versions, avoiding historical version parsing.
- 2026-11-14: Direct PyPI metadata checks for the remaining requested pins (joblib, matplotlib, seaborn) succeeded, resolving the auxiliary-check issue.
- 2026-11-14: Installed the uv bootstrap under `.tools/`; downloading Python 3.13 into `.python/`. Both paths are excluded through `.git/info/exclude`. Added local VS Code settings selecting `.venv/bin/python`, disabling Pylance indexing, and excluding environments/generated output from watching and searching.
- 2026-11-14: Python 3.13.18 installed successfully and `.venv` was recreated with it. Dependency resolution and downloads succeeded with every upstream pin unchanged. uv warned that `/tmp` cache files cannot be hardlinked onto the shared filesystem and automatically fell back to copying; use `UV_LINK_MODE=copy` for future installations.
- 2026-11-14: Added `util/check_environment.py` for reproducible training-only validation and `SETUP.md` for cluster activation/recreation instructions. `.python-version` records Python 3.13.18.
- 2026-11-14: The Python-version installation failure is resolved: Python 3.13.18 installs every pinned dependency and completes the full environment smoke check successfully.

- 2026-11-15: Initial `git status` in `DRUG_CL` failed because the repository is in `DRUG_CL/drug_cl`; using that project root resolves the error. A scoped search found no additional AGENTS.md (rg returned 1 for no matches).
- 2026-11-15: Browser access to the GitHub branches page failed with a restricted-URL error. Direct `git ls-remote` and `git fetch origin` succeeded and provide the branch contents for review.
- 2026-11-15: Remote `knn` is `d2612fe`, `xgboost` is `7ead16c`, and latest `main` is `5f1940d`; main already merges both models and moves the XGBoost grid into YAML. Local uncommitted LR work was based on `4127846` and will be preserved on `RL` before rebasing onto latest main.
- 2026-11-15: Current host `n121` is allocated to this user by running Slurm job `57878551` (`dept_cpu`, `vscode-cpu`). Run model validation through this allocation with one CPU and BLAS/OpenMP thread limits of one.
- 2026-11-15: Preserved all existing LR work in branch `RL` and rebased it cleanly onto `origin/main` (`5f1940d`). User clarified: create/push this branch only; do not merge it into main or open a PR. Remote main is untouched.
- 2026-11-15: Reviewed model conventions: kNN uses 20 YAML configurations (neighbors, metric, voting weights) with scaling; XGBoost uses 48 configurations without scaling and resolves positive-class weights from each training fold. LR uses the same preprocessing, five scaffold folds, seven metrics, ROC-AUC selection, and pipeline persistence, with 14 L2/class-weight configurations and scaling. The upstream tuning harness, model entry points, and kNN/XGBoost CSVs are unchanged on `RL`.
- 2026-11-15: Initial post-rebase LR validation completed on Slurm job `57878551`, exit code 0: 70 fits in 8.3 seconds with no convergence warnings. Both LR result CSVs exactly reproduce the previously committed files. Log: `logs/logistic_regression_branch_validation.log`.
- 2026-11-15: Aligned the LR entry point with XGBoost's `MODEL_NAME` convention and explicit keyword arguments to the common tuning API. Added a three-model comparison in SETUP.md documenting shared behavior and necessary model-specific differences. User confirmed that the finished work should be committed and pushed as a branch only.
- 2026-11-15: Compatibility check passed exact fold equality, all 70 LR fits, reloaded winning metrics, fold-only scaling, and all seven metrics for one kNN configuration across five folds. It failed the XGBoost comparison against the repository CSV (maximum metric difference 0.13043478). Upstream XGBoost entry point, tuning code, preprocessing, requirements, and data are byte-identical on `RL`. Investigating by executing the original upstream grouping/tuning code in the same environment; the upstream CSV is preserved. Log: `logs/branch_compatibility_check.log`.
- 2026-11-15: Branch compatibility concern resolved: loaded original grouping/tuning source from `origin/main` and reran the same XGBoost configuration in the same allocation. Its five-fold results are bit-for-bit identical to `RL`, including all seven metrics; per-fold class weights and absence of scaling also pass. The historical CSV discrepancy reproduces with unchanged upstream code, so its root cause remains an upstream reproducibility question, not an effect of this branch. No XGBoost results were overwritten. Log: `logs/xgboost_compatibility_check.log`; initial failed comparison preserved in `logs/branch_compatibility_check_initial.log`.
- 2026-11-15: Final LR entry-point validation passed all 70 fits and exactly reproduced existing CSVs. Verified persisted folds against upstream splitter, all seven winning-model metrics after reload, scaler training-fold statistics, kNN representative metrics, and XGBoost same-environment compatibility. All validation data access was guarded to reject the held-out test split. `git diff --check` passed.
- 2026-11-15: Created remote branch `RL` and verified its implementation commit `33abdc3f2761c7e1f133986e37b9331f1b1cceab` with `git ls-remote`. Remote main remains `5f1940ddf5bb7bb4485d310d2b9f12103f625a3a`. No merge or pull request was created. Branch: https://github.com/amyji-ui/drug-induced-autoimmunity-ml/tree/RL . This final documentation update records the verified publication.
