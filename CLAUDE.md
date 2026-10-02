# induction-heads-compact-proofs

Compact proofs of accuracy for a small 2-layer attention-only transformer trained on an induction task. See README.md for the task, method and results.

## Layout

- `src/` flat Python scripts, run from the repo root as `python src/<script>.py ...`. Scripts import each other by module name (e.g. `from proof_v1 import build`), which works because Python puts `src/` on the path when running a script there.
- `weights/` trained parameters (`params_s{seed}.pkl` for n=6, `params_n8_s{seed}.pkl` for n=8). Pickled dicts of numpy arrays.
- `results/` logs, one file per seed: `res_*` (brute force, v3 bound, soundness), `v3_rigorous_*`, `v4_*` (v4 float and rigorous, soundness, float-vs-rigorous comparison).
- `tests/` checks of the interval arithmetic against exact rational arithmetic, and of each rigorous bound against its float version.

## Conventions

- Task size comes from env vars `D` (vocab, default 16) and `N` (length, default 6), read in `src/model.py`. Use `N=8` for the length-8 run.
- `proof_v4.py` is the current bound. Earlier versions are kept because later ones import from them (`build` from v1, `Group` and the LP helpers from v2, `row_box` from v3).
- Each of v3 and v4 has two implementations: `proof_vN.py` in plain float64 (fast, readable, sound only in exact arithmetic) and `proof_vN_rigorous.py` in outward-rounded interval arithmetic (`interval.py`). Only the rigorous one is a certificate. A change to a bound goes into both, and `tests/check_rigorous_vs_float.py` must show they agree to rounding error.
- `interval.py` may only rely on single correctly-rounded numpy operations (add, subtract, multiply, divide, sqrt) and `np.sum`; no BLAS, `einsum`, `np.exp` or other libm calls on the certified path. New primitives need a test against exact arithmetic in `tests/test_interval.py`.
- In rigorous code, names ending `_lo` / `_hi` are one-sided float bounds; `Iv` objects are two-sided enclosures.
- Any change to a bound must keep `src/sound.py` at 0 violations (it takes the proof module name as second argument).
- The certified statement is about the real-arithmetic function defined by the float64 weights with a hard causal mask, not about a particular floating-point forward pass.

## Environment

`uv venv && uv pip install -r requirements.txt` (numpy, autograd). The local venv is `.venv/`.
