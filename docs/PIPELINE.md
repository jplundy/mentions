# Historical Transcript Data Pipeline

This repository contains a configurable pipeline for assembling a historical corpus of
FOMC press conferences and corporate earnings call transcripts. The workflow
implements the PM1 responsibilities described in `TODO.txt` and produces a structured
Parquet dataset for downstream modeling teams.

## Overview

1. **Inventory management** – Provide a CSV or JSON file that enumerates all transcripts
   with metadata (event id, type, date, speakers, PDF filename, source URL, provenance).
2. **PDF ingestion** – Text is extracted from each PDF using `pdfplumber` (with an
   automatic fallback to `pypdf`).
3. **Preprocessing** – Headers/footers are removed, speaker labels normalized, and
   Unicode punctuation harmonized.
4. **Segmentation** – Cleaned transcripts are split into speaker turns by default, or
   into fixed token-length windows if configured.
5. **Dataset publishing** – Segments are exported to Parquet with metadata, boolean
   target-word labels, validation summaries, and manifest files.

## Configuration

Pipeline execution is driven by a YAML or JSON configuration file. A sample file is
included at `configs/example_pipeline.yaml`, and a CSV template is available at
`data/inventory_template.csv`. Key fields:

- `inventory_path`: CSV/JSON inventory describing transcripts (must include a
  `pdf_file` column and `event_id`). Paths are resolved relative to the config file.
- `pdf_directory`: Directory containing the PDF files referenced by the inventory.
- `output_directory`: Where Parquet datasets and manifests will be written.
- The pipeline validates that each referenced PDF exists before processing.
- `target_words`: List of words/phrases to track; boolean label columns named
  `target__<word>` are added for each entry.
- `segmentation`: Controls the segmentation strategy; `speaker_turn` uses detected
  speaker labels, while `fixed_window` requires a `window_size` (and optional `stride`).
- `speaker_filter`: Optional include/exclude lists for restricting segments to
  specific speakers. Terms are matched case-insensitively; prefix with
  `regex:` to provide a custom regular expression.
- `metadata_overrides`: Optional per-event metadata updates applied after loading the
  inventory.

## Running the pipeline

```bash
python run.py configs/example_pipeline.yaml --version v0
```

The command will:

1. Load the transcript inventory and validate PDF availability.
2. Extract and clean text from each PDF.
3. Segment the text and build labeled rows.
4. Validate that segments are unique and non-empty.
5. Write `segments_<version>.parquet` and a manifest JSON file capturing validation
   results, column mappings, and metadata. Log each release in `docs/CHANGELOG.md`.

## Data schema

| Column            | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `event_id`        | Unique identifier for the event                      |
| `segment_id`      | Unique identifier for the segment                    |
| `event_type`      | `fomc_press_conference`, `earnings_call`, etc.       |
| `event_date`      | Event date (ISO format)                              |
| `speaker`         | Normalized speaker label (if detected)               |
| `text`            | Cleaned segment text                                 |
| `start_char`      | Character offset where the segment begins            |
| `end_char`        | Character offset where the segment ends              |
| `source_url`      | Original transcript URL (if available)               |
| `provenance`      | Additional provenance or licensing notes             |
| `target__<word>`  | Boolean indicator if target word appears in segment (word is slugified to be column-safe) |
| `...`             | Additional metadata columns copied from inventory    |

A manifest file is emitted alongside the dataset with validation status and versioning
information to aid reproducibility.

## Adding validation checks

Validation checks can be extended by updating `DatasetPublisher.validate`. Examples
include enforcing minimum segment length, verifying coverage of target words, or
comparing segment counts between versions.

## Tooling notes

- Install `pdfplumber` or `pypdf` to enable PDF extraction.
- Install `pandas` (preferred) or `pyarrow` for Parquet writes.
- Unit tests can be added with fixtures that generate lightweight PDFs containing
  synthetic transcripts for deterministic regression testing.
