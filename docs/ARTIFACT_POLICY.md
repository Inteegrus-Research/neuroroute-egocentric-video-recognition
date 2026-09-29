# Artifact Policy

The repository tracks lightweight scientific evidence directly:

- CSV
- JSON
- Markdown
- PNG
- small configuration/manifests

Large tensor/checkpoint files are not committed to ordinary Git by
default.

Before publishing a checkpoint, inspect its size. Files exceeding
GitHub's ordinary single-file limit should use Git LFS or an external
artifact release rather than normal Git history.

The raw EgoBrain dataset and extracted per-subject feature tensors are
not redistributed by this repository.
