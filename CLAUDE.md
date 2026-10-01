# induction-heads-compact-proofs

Compact proofs of accuracy for a small 2-layer attention-only transformer trained on an induction task. See README.md for the task, method and results.

## Layout

- `src/` flat Python scripts, run from the repo root as `python src/<script>.py ...`. Scripts import each other by module name (e.g. `from proof_v1 import build`), which works because Python puts `src/` on the path when running a script there.
- `weights/` trained parameters (`params_s{seed}.pkl` for n=6, `params_n8_s{seed}.pkl` for n=8). Pickled dicts of numpy arrays.
- `results/` logs: brute-force accuracy, certified bound, soundness check, one file per seed.

## Conventions

- Task size comes from env vars `D` (vocab, default 16) and `N` (length, default 6), read in `src/model.py`. Use `N=8` for the length-8 run.
- `proof_v3.py` is the current bound; `proof_v1.py` and `proof_v2.py` are kept because v3 imports `build`, `Group` and the LP helpers from them.
- Any change to a bound must keep `src/sound.py` at 0 violations.
- Bounds are sound in exact arithmetic only; floating-point rounding is not controlled.

## Environment

`uv venv && uv pip install -r requirements.txt` (numpy, autograd). The local venv is `.venv/`.
