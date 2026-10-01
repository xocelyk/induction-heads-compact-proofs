# Papercuts

Small frictions hit while working in this repo.

**2026-10-01** (claude-opus-5-5): Importing the original code from an archive: `model.py` saved weights to a hardcoded absolute path from the environment it was written in, so training failed elsewhere. Now saves to `weights/` relative to the script.
