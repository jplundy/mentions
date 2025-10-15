# Integrating Pre-Meeting News Signals into Powell Mention Forecasts

This note extends the Powell FOMC workflow by outlining how to incorporate
pre-meeting news coverage—such as Wall Street Journal (WSJ) previews—into the
probability forecasts that Jerome Powell will utter a tracked phrase during the
press conference. The goal is to translate qualitative reporting into structured
features that modify the base transcript-driven model.

## 1. High-level architecture

1. **Baseline transcript model** – Existing pipeline trained on historical press
   conference transcripts produces calibrated probabilities for each
   target phrase per segment.
2. **News signal ingestion** – Fetch and store recent articles discussing the
   upcoming FOMC meeting. Restrict to trusted outlets (WSJ, Bloomberg, Fed
   previews) and normalize their metadata (publication time, authors,
   headlines, URLs).
3. **News feature builder** – Convert article text into numeric signals aligned
   with the model's target vocabulary (e.g., counts of the tracked phrases,
   contextual embeddings, sentiment scores). Aggregate signals to the
   meeting level.
4. **Adjustment layer** – Blend the news-derived signals with the baseline
   model. Options include:
   - augmenting the feature matrix and retraining a joint model;
   - post-processing baseline logits with a learned regression on news
     features (Bayesian updating);
   - scenario analysis that shifts predicted probabilities according to
     rule-based heuristics when strong news cues are present.
5. **Monitoring & override** – Track how much the adjustments move the final
   probabilities, log provenance for each news article, and provide manual
   override hooks when editors need to enforce or veto adjustments.

## 2. News ingestion workflow

1. **Source discovery**
   - Maintain a YAML/CSV catalog of RSS feeds or API endpoints to poll for
     FOMC-related coverage.
   - Include metadata describing access methods (API key, scraping, manual
     download) and licensing caveats.
2. **Acquisition cadence**
   - Run a daily job in the week leading up to the meeting; increase to
     twice daily in the final 48 hours.
   - De-duplicate articles by normalized URL or hashed headline + timestamp.
3. **Storage format**
   - Persist raw articles as JSONL in `data/news/raw/` with fields:
     `article_id`, `source`, `published_at`, `headline`, `url`, `body_text`,
     `tags`.
   - Track retrieval metadata (HTTP status, scraper version) for auditing.
4. **Preprocessing**
   - Clean HTML, remove boilerplate, standardize quotes/hyphens, and record
     token counts.
   - Optionally summarize long articles using extractive techniques to
     reduce noise before feature extraction.

## 3. Feature engineering blueprint

1. **Target-word proximity**
   - Measure how often each tracked phrase appears in the article body or
     headline.
   - Compute TF–IDF scores for windows around each target phrase to capture
     contextual hints (e.g., "credit tightening", "balance sheet").
2. **Topic relevance**
   - Use a pretrained sentence embedding model (e.g., `all-MiniLM-L6-v2`) to
     embed paragraphs mentioning "Federal Reserve", "FOMC", or "Powell".
   - Cluster or classify these embeddings into themes (inflation, labor
     market, banking stress) and produce theme intensity scores.
3. **Sentiment & modality**
   - Apply finance-tuned sentiment lexicons to quantify tone (hawkish vs
     dovish) and highlight conditional language ("may signal", "expected to").
4. **Temporal decay**
   - Weight article contributions using an exponential decay from publication
     time to the meeting start (e.g., half-life of 24 hours) to emphasize
     the freshest reporting.
5. **Aggregation granularity**
   - Produce both meeting-level aggregates (sums/means/max) and article-level
     features. Meeting-level vectors will join directly with the dataset of
     transcript segments using `event_id`.
   - Retain article-level features separately for audit trails and potential
     future sequence models.

## 4. Adjustment strategies

1. **Joint re-training**
   - Extend the experiment configuration to include a `news_features_path`
     pointing to aggregated meeting-level features.
   - Update the modeling pipeline to merge these features with the
     transcript-based feature matrix before training and calibration.
2. **Post-hoc logistic adjustment**
   - Train a lightweight regression (e.g., ridge) on historical meetings where
     both transcripts and news features exist. Use the baseline logits as an
     input feature alongside the news signals to predict realized outcomes.
   - At inference, pass the baseline logit through the regression to obtain
     the adjusted probability, preserving interpretability.
3. **Bayesian prior shift**
   - Model the baseline probability as the likelihood and treat news features
     as informative priors on the phrase occurrence rate. Implement via a
     conjugate prior (e.g., Beta-Bernoulli) whose parameters are functions of
     the news signals.
4. **Rule-based overrides**
   - Define deterministic rules for rare but high-impact cues (e.g., if a
     WSJ scoop explicitly quotes Powell's planned remarks). Implement as
     additive or multiplicative adjustments with manual approval.

## 5. Evaluation plan

1. **Backfill dataset**
   - For past FOMC meetings with available news archives, reconstruct the
     article corpus and compute features using the same pipeline.
2. **Offline experiments**
   - Compare baseline vs. news-augmented models on leave-one-meeting-out
     validation. Track log-loss, Brier score, calibration curves, and
     probability shift magnitudes.
   - Run ablations to isolate feature contributions (e.g., embeddings only,
     sentiment only).
3. **Sensitivity analysis**
   - Stress test the adjustments by perturbing news feature values to ensure
     the model does not overreact to single articles.
4. **Human review loop**
   - Present analysts with a dashboard summarizing the news signals and their
     impact on probabilities for at least the last five meetings. Gather
     qualitative feedback to refine heuristics and thresholds.

## 6. Integration tasks

1. Implement a `news_ingest.py` CLI mirroring `run.py` conventions for
   scheduling ingestion jobs.
2. Add a `news_features.py` module that consumes the raw JSONL articles and
   emits Parquet feature tables keyed by `event_id`.
3. Extend `configs/example_experiment.yaml` with optional `news_features_path`
   and update `modeling/dataset.py` to merge auxiliary feature tables when
   configured.
4. Update documentation (`docs/PIPELINE.md`, `docs/MODELING.md`) once the
   prototype stabilizes.
5. Configure experiment tracking to log the news corpus snapshot hash for
   reproducibility and attach article provenance to scored events.

## 7. Open questions & risks

- **Access limitations** – WSJ content is paywalled; confirm licensing and
  consider manual summaries or alternative public sources if APIs are
  inaccessible.
- **Lag vs. freshness** – Determine cutoffs for including very recent
  articles that may arrive after the model run but before the press
  conference.
- **Signal dilution** – Guard against irrelevant articles triggering false
  adjustments. Topic filters and manual vetting may be required in the early
  iterations.
- **Operational load** – Decide whether the news adjustment runs as part of
  the scheduled pipeline or as an analyst-triggered step with review gates.

By following this blueprint we can systematically translate qualitative
pre-meeting reporting into quantifiable modifiers that enhance Powell mention
forecasts while keeping provenance and override controls in place.
