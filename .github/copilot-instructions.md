# Copilot instructions for proxy_prep ✅

Purpose: Help an AI coding agent become immediately productive by highlighting project intent, developer workflows, reproducibility conventions, and concrete examples from the codebase.

---

## Quick project summary
- Small teaching / exercise repo with a focus on small, pure Python functions and short experiments in notebooks.
- Key files:
  - `day1_basics.py` — core utilities (normalize, train_test_split_indices, rmse).  Unit-tested.
  - `test_day1_basics.py` — pytest-based tests and examples of expected behavior.
  - `day2_numpy_lr.ipynb` — experiment/notebook using NumPy and Matplotlib.
  - `notes_day1.md`, `notes_day2.md` — developer notes.

## How to run and test 🔧
- Run the module demo: `python day1_basics.py`  (prints demo outputs)
- Run tests: `python -m pytest -q` (pytest is the canonical test runner)
- Run a single test: `pytest test_day1_basics.py::test_name -q`
- Notebooks: open `day2_numpy_lr.ipynb` in VS Code/Jupyter; it uses `numpy` and `matplotlib` and expects an environment with those packages.

## Project-specific conventions and gotchas ⚠️
- Determinism:
  - Use deterministic RNGs for reproducible behavior. Example: `train_test_split_indices` uses `random.Random(seed)` (pure Python RNG) and seeds must reproduce splits.
  - Notebooks use `np.random.default_rng(seed)` in experiments (`day2_numpy_lr.ipynb`).
- Test split sizing:
  - `test_size = int(round(n * test_ratio))` then clamped to `[1, n-1]`. Preserve this rounding-and-clamp behavior when modifying splitting logic.
- Normalization:
  - `normalize` uses *population* std (divide by n, not n-1). For constant inputs it returns a list of zeros — keep this edge-case behavior.
- Errors & validation:
  - Functions validate inputs and raise `ValueError` for invalid cases (see `rmse` length mismatch and empty checks). Follow this pattern.
- Typing and docstrings: Prefer small functions with type hints and clear docstrings.

## Concrete examples to reference
- Keep these behaviors intact or update tests when changing them:
  - `day1_basics.py::normalize(xs)` — returns [] for empty input, zeros for constant input, unit variance using population std.
  - `day1_basics.py::train_test_split_indices(n, test_ratio, seed)` — deterministic with `seed`, uses shuffle from `random.Random`, rounding+clamping for test size.
  - `day1_basics.py::rmse(y_true, y_pred)` — raises on mismatched lengths and empty input.

## Typical tasks an AI agent will be asked to do 🧭
- Add targeted unit tests for edge cases (e.g., extreme ratios, small n, non-numeric inputs where appropriate).
- Implement algorithmic fixes while preserving existing documented behavior (update tests if behavior changes intentionally).
- Small refactors to improve readability or add type hints; ensure all tests still pass.
- Make reproducible notebook experiments by specifying RNG seeds and documenting plotting steps in `day2_numpy_lr.ipynb`.

## What to avoid
- Do not change deterministic behavior (seed handling, RNG type) without a corresponding test that intentionally updates the contract.
- Avoid switching from population to sample std silently—this changes semantics; update tests and docstrings if you intend to change it.

---

If anything in this summary is unclear or you'd like additional project-specific rules (e.g., formatting, linting, CI or packaging steps), tell me what to add and I will update this file. 🙌
