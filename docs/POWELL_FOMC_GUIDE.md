# Predicting Jerome Powell Mentions in FOMC Press Conferences

This guide walks through the end-to-end workflow for estimating the probability that
Chair Jerome Powell says a given word or phrase during an FOMC press conference using
this repository's historical transcript pipeline and baseline modeling stack.

The process has three stages:

1. Build a structured dataset of past FOMC press conference transcripts with Powell
   labels.
2. Train and calibrate a baseline predictive model on the labeled segments.
3. Score new or hypothetical transcript segments to produce probabilities that Powell
   will utter the tracked phrase.

## 1. Environment setup

1. Use Python 3.9+ and create an isolated environment (e.g., `python -m venv .venv`).
2. Activate the environment and install dependencies required by the pipeline and
   modeling code:

   ```bash
   source .venv/bin/activate
   pip install pandas pyarrow scikit-learn pdfplumber pypdf sentence-transformers joblib
   ```

   - `pdfplumber`/`pypdf` enable transcript ingestion.
   - `pandas`/`pyarrow` power dataset loading and Parquet export.
   - `scikit-learn`, `sentence-transformers`, and `joblib` support feature extraction,
     modeling, calibration, and artifact persistence.

3. Verify the repository structure matches the documented layout in
   `docs/PIPELINE.md` and `docs/MODELING.md` before proceeding.

## 2. Assemble the Powell-focused transcript inventory

1. Collect PDF transcripts for past FOMC press conferences chaired by Jerome Powell.
   Include the official FOMC press conference PDF, the event date, and any metadata
   you plan to use as features (e.g., speaker list, event type).
2. Copy `data/inventory_template.csv` to a new file (for example,
   `data/powell_fomc_inventory.csv`). Fill in one row per press conference with:

   - `event_id`: a unique slug (e.g., `fomc_2023_06_powell`).
   - `event_type`: `fomc_press_conference`.
   - `event_date`: ISO-format date of the conference.
   - `speakers`: include `Jerome Powell` and other speakers if needed.
   - `pdf_file`: filename of the downloaded transcript PDF stored in
     `data/pdfs/` (create the directory if it does not exist).
   - `source_url` and `provenance`: optional but recommended for traceability.

   The pipeline expects all PDF files referenced in the inventory to exist relative to
   the `pdf_directory` path you supply in the configuration.

3. Confirm the inventory loads without errors by opening it in a spreadsheet or using
   `pandas.read_csv` in a Python shell.

## 3. Configure and run the historical pipeline

1. Duplicate `configs/example_pipeline.yaml` and save it as
   `configs/powell_fomc_pipeline.yaml`.
2. Update the configuration with Powell-specific settings:

   ```yaml
   inventory_path: ../data/powell_fomc_inventory.csv
   pdf_directory: ../data/pdfs
   output_directory: ../data/derivatives/powell_fomc
   target_words:
     - powell
     - "chair powell"
   segmentation:
     mode: speaker_turn
   ```

   - Add as many phrases to `target_words` as you wish to monitor. The pipeline will
     create boolean columns named `target__<slug>` for each entry.
   - Use the `metadata_overrides` block if you need to correct speaker names or add
     additional columns before modeling.

3. Run the pipeline from the repository root:

   ```bash
   python run.py configs/powell_fomc_pipeline.yaml --version powell_v1
   ```

   The CLI loads the configuration, extracts text from each PDF, preprocesses the
   transcript, segments it by speaker turn, and emits a labeled Parquet dataset along
   with validation metadata.【F:run.py†L1-L70】【F:docs/PIPELINE.md†L1-L65】

4. Inspect the outputs in `data/derivatives/powell_fomc/`. You should see:

   - `segments_powell_v1.parquet`: cleaned segments with metadata, Powell target
     columns (e.g., `target__powell`), and validation flags.
   - `manifest_powell_v1.json`: contains validation summaries for reproducibility.

## 4. Configure a modeling experiment for Powell probabilities

1. Copy `configs/example_experiment.yaml` to
   `configs/powell_fomc_experiment.yaml` and edit the key fields:

   ```yaml
   dataset_path: ../data/derivatives/powell_fomc/segments_powell_v1.parquet
   target_column: target__powell
   feature:
     text:
       column: text
       params:
         max_features: 20000
         ngram_range: [1, 2]
         min_df: 2
         max_df: 0.95
     categorical:
       columns: [event_type, speaker]
       drop: null
       sparse: true
     include_embeddings: false
   model:
     type: logistic_regression
     params:
       C: 1.0
   validation:
     strategy: leave_one_event_out
     minimum_train_events: 3
   calibration:
     method: isotonic
     cv: 3
   tracking:
     output_directory: experiments/powell_baseline
     use_mlflow: false
   ```

   - The TF–IDF and categorical feature blocks mirror the defaults used by the
     baseline pipeline documented in `docs/MODELING.md`.
   - Leave-one-event-out validation guards against temporal leakage between press
     conferences, and isotonic calibration improves probability calibration across
     folds.【F:docs/MODELING.md†L1-L78】【F:configs/example_experiment.yaml†L1-L26】

2. Ensure the `minimum_train_events` setting is less than the total number of Powell
   press conferences in your dataset; otherwise, adjust the value or supply more
   historical transcripts.

## 5. Train, evaluate, and calibrate the baseline model

1. Execute the modeling CLI:

   ```bash
   python model.py configs/powell_fomc_experiment.yaml --output experiments/powell_baseline
   ```

   - The script loads the Parquet dataset, builds the configured feature pipeline, and
     runs leave-one-event-out validation for each press conference.【F:model.py†L1-L33】
   - Metrics such as log-loss, Brier score, ROC-AUC, precision, recall, and accuracy
     are printed to stdout and stored in the tracking directory via the
     `ExperimentTracker` utilities.【F:modeling/experiments.py†L1-L152】

2. Review the artifacts written to `experiments/powell_baseline/`:

   - `config.yaml`: resolved experiment configuration.
   - `fold_<k>_metrics.json`: per-fold validation metrics.
   - `summary.json`: aggregate metrics and calibration scores.
   - `baseline_pipeline.joblib`: the calibrated pipeline ready for inference.
   - `feature_importances.json`: interpretable feature scores when available.

3. Use the calibration report (Brier score, log-loss) to assess whether additional
   features (e.g., embeddings or macro indicators) are needed before deployment.

## 6. Score new Powell FOMC scenarios

1. Load the trained pipeline in a Python session:

   ```python
   import joblib
   import pandas as pd

   pipeline = joblib.load("experiments/powell_baseline/baseline_pipeline.joblib")
   ```

2. Prepare a DataFrame of segments you want to evaluate. For a forthcoming press
   conference, you can craft hypothetical segments or transcribe remarks from a
   rehearsal and supply required metadata columns (`event_id`, `event_type`,
   `speaker`, `text`, etc.). Ensure the column names match those used during training.

   ```python
   candidate_segments = pd.DataFrame([
       {
           "event_id": "fomc_2024_03_preview",
           "event_type": "fomc_press_conference",
           "speaker": "Jerome Powell",
           "text": "We remain committed to restoring price stability while monitoring the labor market.",
       }
   ])
   ```

3. Generate calibrated probabilities that Powell will mention the tracked phrase in
   each segment:

   ```python
   probabilities = pipeline.predict_proba(candidate_segments)[:, 1]
   print(probabilities)
   ```

4. Aggregate segment-level probabilities to produce event-level insights (e.g., mean
   probability, maximum probability, or custom aggregation logic) for prediction market
   inputs.

5. When fresh transcript data becomes available after a press conference, append it to
   your dataset, rerun the historical pipeline with a new `--version`, and retrain or
   recalibrate the model to keep probabilities current.

## 7. Next steps and enhancements

- Incorporate `SentenceTransformerBlock` embeddings by setting
  `include_embeddings: true` in the experiment config for richer semantic context.
- Extend the pipeline's `target_words` list with additional phrases (e.g., "inflation"
  or "balance sheet") to build a multi-label Powell probability dashboard.
- Integrate current-event modifiers (macro data, sentiment indices) via custom feature
  blocks as outlined in `docs/MODELING.md` and the PM3 milestones in `TODO.txt`.

Following this process produces a reproducible, validated baseline for forecasting the
likelihood that Chair Powell utters specific phrases during FOMC press conferences.
