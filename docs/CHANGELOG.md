# Data Pipeline Change Log

This change log tracks updates to the historical transcript dataset and preprocessing
pipeline. Record each run with the dataset version, configuration checksum, and summary
of changes.

| Date       | Dataset Version | Config Path                 | Summary                                   |
| ---------- | ---------------- | --------------------------- | ----------------------------------------- |
| YYYY-MM-DD | v0               | configs/example_pipeline.yaml | Initial dataset assembly pipeline created |

Guidelines:

- Increment the dataset version (e.g., `v1`, `v1.1`) whenever the inventory, target words,
  or preprocessing heuristics change.
- Attach validation summaries generated in `data/processed/segments_<version>.json`.
- Note any manual corrections or overrides applied to transcripts.
