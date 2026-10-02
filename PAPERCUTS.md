# Papercuts

Small frictions hit while working in this repo.

**2026-10-01** (claude-opus-5-5): Importing the original code from an archive: `model.py` saved weights to a hardcoded absolute path from the environment it was written in, so training failed elsewhere. Now saves to `weights/` relative to the script.

**2026-10-02** (claude-fable-5-1): Changed the default module of `src/sound.py` and `tests/check_rigorous_vs_float.py` while a background loop over seeds was still calling them without the argument, so the last seed's log compared a different version than the others and had to be rerun. Pass the proof module explicitly in batch runs.
