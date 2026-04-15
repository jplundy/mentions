# Mentions: Codebase Analysis & Improvement Plan

## Current State Assessment

### What This Project Does
A prediction market edge-finding system. It ingests historical transcripts
(FOMC press conferences, NFLX earnings calls), builds probabilistic models
to predict whether specific words/phrases will be spoken, and compares those
probabilities against live Kalshi market odds to identify mispriced contracts.

### Architecture (3 modules)
```
pipeline/    -> PDF ingestion, preprocessing, speaker filtering, segmentation, Parquet publishing
modeling/    -> TF-IDF features, logistic regression/GBM, time-aware validation, calibration, tracking
news/        -> External article ingestion, news-derived feature generation
```
Supporting: `dash_app.py` (explorer UI), `KalshiClient` (market data), experiment configs (YAML).

### Current Metrics (powell_baseline4 - latest)
| Metric            | Value  | Assessment          |
|-------------------|--------|---------------------|
| Accuracy          | 96.0%  | Misleading (imbalance) |
| F1                | 0.0    | **Model predicts all negatives** |
| Precision         | 0.0    | No positive predictions |
| Recall            | 0.0    | Misses every positive |
| Log Loss          | 0.241  | Decent for calibration |
| Brier             | 0.053  | Low but trivially achievable |
| Avg Precision     | 0.234  | Weak ranking ability |
| ROC AUC           | NaN    | Broken (single-class folds) |

**The model has zero predictive power on the positive class.** It achieves
high accuracy by always predicting "no" on a ~96/4 class split. This is the
single most critical problem.

---

## Improvement Plan: Prioritized by Edge Impact

### TIER 1 - Fix the Broken Foundation (Immediate)

#### 1.1 Class Imbalance Handling
**Problem:** ~4% positive rate means the model learns to always predict negative.
**Fix:**
- Add `class_weight="balanced"` to logistic regression config
- Implement SMOTE/ADASYN oversampling as a pipeline step option
- Add configurable decision threshold tuning (optimize for F1 or custom utility)
- Track class distribution in experiment metadata

**Files:** `modeling/models.py`, `modeling/config.py`, `modeling/experiments.py`

#### 1.2 Fix ROC AUC NaN
**Problem:** Leave-one-event-out folds can produce single-class validation sets.
**Fix:**
- Add guard in `compute_classification_metrics()` to handle single-class folds
- Skip ROC AUC when only one class present, log a warning
- Add minimum positive count threshold to validation splits
- Consider stratified grouping to ensure both classes in each fold

**Files:** `modeling/evaluation.py`, `modeling/validation.py`

#### 1.3 Decision Threshold Optimization
**Problem:** Hard 0.5 threshold is wrong for imbalanced data.
**Fix:**
- Sweep thresholds on validation folds, select by max F1 or expected profit
- Persist optimal threshold alongside the model artifact
- Use threshold in inference pipeline

**Files:** `modeling/evaluation.py`, `modeling/experiments.py`

---

### TIER 2 - Feature Engineering for Edge (High Impact)

#### 2.1 Activate Sentence Transformer Embeddings
**Status:** Code exists (`SentenceTransformerBlock`) but `include_embeddings: false` in all configs.
**Fix:**
- Run experiments with embeddings enabled; compare against TF-IDF alone
- Test ensemble (TF-IDF + embeddings concatenated)
- Benchmark `all-MiniLM-L6-v2` vs `all-mpnet-base-v2` vs domain-tuned models
- Add dimensionality reduction (PCA/UMAP) option for embedding features

**Files:** configs, `modeling/pipeline.py`, `modeling/features/text.py`

#### 2.2 Positional & Structural Features
**Problem:** Current features are bag-of-words; ignores WHERE in the transcript a word appears.
**Fix:**
- Add segment position features: `segment_index / total_segments`, early/mid/late bins
- Speaker transition features: who spoke before/after, topic shift signals
- Segment length, question vs. statement detection
- Time-into-event features (if timestamps available)

**Files:** New `modeling/features/structural.py`

#### 2.3 Historical Frequency Features
**Problem:** No cross-event learning about target word frequency trends.
**Fix:**
- Rolling frequency of target word in past N events
- Speaker-specific word frequency history
- Trend direction (increasing/decreasing usage over time)
- "Surprise" score: deviation from historical baseline

**Files:** New `modeling/features/historical.py`

#### 2.4 News Feature Integration (Complete the Pipeline)
**Status:** `NewsFeatureBuilder` exists but isn't wired into experiment configs.
**Fix:**
- Add `news_features_path` to powell/nflx experiment configs
- Build news features for existing events
- Add sentiment scoring (not just keyword counts) using a lightweight model
- Add news recency features: hours between latest article and event

**Files:** configs, `news/news_features.py`

---

### TIER 3 - Model Architecture Upgrades (Medium Impact)

#### 3.1 Gradient Boosting with Proper Tuning
**Status:** Config supports `gradient_boosting` but only logistic regression is used.
**Fix:**
- Add XGBoost/LightGBM as model options (better with sparse features + imbalance)
- Implement basic hyperparameter search (Optuna or grid search)
- Use `scale_pos_weight` for native imbalance handling in GBM
- Compare against logistic regression baseline

**Files:** `modeling/models.py`, `modeling/config.py`

#### 3.2 Model Ensembling
**Problem:** Single model is fragile; ensembles are more robust for betting.
**Fix:**
- Implement stacking: TF-IDF logistic + embeddings GBM + news features
- Add simple averaging/weighted blending of fold models
- Per-target-word model selection (some words may need different architectures)

**Files:** New `modeling/ensemble.py`

#### 3.3 Bayesian / Probability-Native Models
**Problem:** Current calibration is post-hoc (isotonic); native probabilistic models are better.
**Fix:**
- Add Bayesian logistic regression option (informative priors from historical rates)
- Implement beta-binomial model for low-data targets
- Add confidence intervals to predictions (essential for bet sizing)

**Files:** New `modeling/bayesian.py`

---

### TIER 4 - Live Inference & Market Integration (The Edge)

#### 4.1 Real-Time Scoring Pipeline
**Status:** No inference workflow exists. `model.py` has stale config classes.
**Fix:**
- Build `inference/` module: load trained pipeline, accept raw text + context, emit probabilities
- CLI: `python -m mentions.inference --model experiments/powell_baseline4 --text "..."`
- Accept partial transcript (streaming during live event)
- Output: probability, confidence interval, recommended position

**Files:** New `inference/` module

#### 4.2 Market Comparison & Bet Sizing
**Status:** `KalshiClient` + `compare_model_to_market_odds()` exist but aren't integrated.
**Fix:**
- Wire model output into `MarketComparison`
- Implement Kelly criterion bet sizing given edge and bankroll
- Add configurable edge thresholds (only signal when edge > X%)
- Historical backtest: what would PnL have been on past events?

**Files:** New `inference/strategy.py`, `modeling/markets.py`

#### 4.3 Live Event Monitoring
**Problem:** No way to run the model during an actual press conference.
**Fix:**
- Streaming ingestion: accept text chunks as transcript unfolds
- Update probability in real-time as new segments arrive
- WebSocket or polling integration with the Dash app
- Alert system: notify when probability crosses a threshold

**Files:** New `inference/streaming.py`, `dash_app.py` updates

---

### TIER 5 - Infrastructure & Reliability

#### 5.1 Dependency Management
**Problem:** No `requirements.txt` or `pyproject.toml`. Reproducibility is broken.
**Fix:**
- Create `pyproject.toml` with pinned dependencies
- Separate extras: `[dev]`, `[embeddings]`, `[mlflow]`
- Add `uv.lock` or `pip-compile` for deterministic installs

#### 5.2 Test Coverage
**Status:** 8 test files exist but coverage is shallow; no integration tests.
**Fix:**
- Add end-to-end pipeline test (ingest -> segment -> model -> predict)
- Add property-based tests for pattern matching (`_compile_target_pattern`)
- Mock Kalshi API tests for market comparison
- Target 80%+ coverage on `modeling/` and `pipeline/`
- Add `pytest-cov` and CI gate

#### 5.3 CI/CD Pipeline
**Problem:** None exists.
**Fix:**
- GitHub Actions: lint (ruff), type check (mypy), test (pytest)
- Automated experiment regression: ensure new code doesn't degrade metrics
- Pre-commit hooks for formatting

#### 5.4 Code Cleanup
- `model.py` (root) has stale/duplicate config classes that overlap with `modeling/config.py`
  and `baseline_run.py` — consolidate or remove
- `baseline_run.py` duplicates functionality in `modeling/experiments.py` — deprecate
- `dash_app.py` is 738 lines — extract callbacks into separate modules
- Standardize entry points under a `__main__.py` or CLI group

---

### TIER 6 - Advanced Edge Techniques

#### 6.1 Market Microstructure Features
- Kalshi order book depth / bid-ask spread as features
- Historical candlestick patterns from `kalshi_history.py` as context
- Market price movement momentum (are odds trending?)

#### 6.2 Multi-Target Joint Modeling
- Current setup trains one model per target word independently
- Joint model could share representations across related targets
- "If they say X, they're more likely to say Y" correlations

#### 6.3 Transfer Learning Across Event Types
- FOMC and earnings calls share linguistic patterns
- Fine-tune a shared base model, then specialize per event type
- Leverage the larger corpus for better feature learning

#### 6.4 Calibration Monitoring & Drift Detection
- Track calibration curves over time (not just aggregate)
- Detect when model confidence diverges from realized outcomes
- Auto-retrain trigger when drift exceeds threshold

#### 6.5 Alternative Data Sources
- Fed minutes / dot plot data as structured features
- Social media sentiment (Twitter/X financial accounts)
- Economic indicator releases (CPI, jobs report) aligned to event dates
- Congressional hearing transcripts for political context

---

## Execution Priority Matrix

| Priority | Item | Effort | Edge Impact | Risk if Skipped |
|----------|------|--------|-------------|-----------------|
| P0 | 1.1 Class imbalance | Small | **Critical** | Model is useless |
| P0 | 1.2 Fix ROC AUC NaN | Small | Medium | Blind to ranking quality |
| P0 | 1.3 Threshold optimization | Small | High | Leaving money on table |
| P1 | 2.1 Embeddings | Medium | High | Missing semantic signal |
| P1 | 3.1 GBM tuning | Medium | High | Underfit on interactions |
| P1 | 4.1 Inference pipeline | Medium | **Critical** | Can't trade without it |
| P1 | 5.1 Dependencies | Small | N/A | Can't reproduce |
| P2 | 2.2 Structural features | Medium | Medium | Missing positional signal |
| P2 | 2.3 Historical features | Medium | High | Missing trend signal |
| P2 | 4.2 Bet sizing | Medium | **Critical** | Edge without sizing = ruin |
| P2 | 2.4 News integration | Small | Medium | Missing context signal |
| P3 | 3.2 Ensembling | Large | Medium | Single-model fragility |
| P3 | 4.3 Live monitoring | Large | High | Manual process |
| P3 | 5.2-5.4 Tests/CI/cleanup | Medium | N/A | Tech debt compounds |
| P4 | 6.x Advanced | Large | Variable | Diminishing returns |

## Recommended Sprint Plan

**Sprint 1 (This Week):** Fix P0 items. Class imbalance + threshold tuning
will go from 0% recall to actual predictions. This alone may surface edge.

**Sprint 2 (Next Week):** Activate embeddings, try GBM, build inference CLI.
Get to a state where you can actually score a live event.

**Sprint 3:** Bet sizing + market comparison integration. Historical backtest
to validate edge before risking capital.

**Sprint 4:** Structural features, news integration, monitoring. Polish the
pipeline for production use.
