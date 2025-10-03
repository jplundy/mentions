# Current-Events Integration & Deployment Playbook

This guide describes the PM3 responsibilities for enriching the baseline modeling
stack with current-event signals and delivering an operational inference workflow.
It assumes the historical transcript dataset (PM1) and calibrated baseline models
(PM2) are available.

## External indicator sourcing

1. **Economic & policy calendars** – Pull scheduled macro releases (FOMC decisions,
   CPI, payrolls) from sources such as St. Louis FRED, Federal Reserve calendars, or
   `tradingeconomics` API. Normalize to UTC dates and include scheduled release time.
2. **Market sentiment** – Capture asset-level sentiment measures (equity/FX/Rate
   futures positioning, implied volatility skews) from vendor CSV feeds or
   alternative datasets (e.g., `quandl`, `alphavantage`). Aggregate to daily values
   and align to market close.
3. **Topical news intensity** – Track keyword counts or topic sentiment by scraping
   RSS/news APIs (`newsapi`, `gdeltn`, vendor feeds). Maintain per-source weights to
   mitigate coverage bias.
4. **Quality gates** – For each feed record availability SLA, licensing limits, and
   refresh cadence. Store raw pulls in `data/current_events/raw/<source>/<date>.parquet`
   with schema `{event_date, indicator_name, value, metadata}`.

## Feature engineering handoff

- Produce a normalized table `data/current_events/features.parquet` keyed by
  `event_id` and `event_date` with one column per indicator or categorical bucket.
- Apply transformations compatible with PM2 feature blocks:
  - Numeric indicators: z-score over a 90-day trailing window, Winsorize at the
    1st/99th percentile, and add missing-value flags.
  - Categorical events (e.g., surprise bins): encode as string categories ready for
    one-hot encoding.
- Share a `current_events_schema.yaml` describing column types, fill strategies, and
  upstream source pointers. Coordinate with PM2 to register the features in
  `modeling/features/metadata.py` or a dedicated block implementing `FeatureBlock`.

## Post-model calibration & drift checks

1. Reserve rolling hold-out windows (e.g., last 3 events per category) for PM3-only
   calibration. Load PM2 artifacts and re-fit isotonic/Platt calibrators on the
   augmented feature space.
2. Track calibration metrics (Brier, log-loss, Expected Calibration Error) per
   evaluation window. Write results to `experiments/<run>/pm3_calibration.json`.
3. Implement a drift notebook/script that compares recent event distributions of key
   indicators versus training folds using Population Stability Index (PSI) or
   Kolmogorov–Smirnov tests. Trigger re-calibration if PSI > 0.2 or metrics degrade
   by >10%.

## Inference workflow design

- **Interface** – Provide both a CLI (`python model.py predict --config ...`) and an
  optional REST microservice (`FastAPI`), sharing a core `InferenceEngine` module
  that loads the serialized pipeline, applies latest current-event features, and
  outputs probability + confidence interval per target word.
- **Context ingestion** – Accept:
  1. Latest transcript draft or prepared remarks text.
  2. Event metadata (event type, scheduled date/time, speakers).
  3. Current-event indicator snapshot (auto-fetched if not supplied).
- **Processing steps**:
  1. Normalize inputs to match training schema (speaker labels, tokenization).
  2. Fetch and join current-event features for the event date (fallback to nearest
     business day). Impute missing values using training medians.
  3. Run the trained pipeline to obtain logits and calibrated probabilities.
  4. Estimate confidence intervals via bootstrapped predictions (e.g., 100 resamples
     over calibration residuals) or analytical variance from logistic regression.
- **Outputs** – Emit JSON with probability per target phrase, CI bounds, contributing
  feature scores, and metadata (model version, calibration timestamp).

## Monitoring & alerting

- **Dashboards** – Use a lightweight stack (e.g., `Metabase`, `Superset`, or `Grafana`)
  to visualize:
  - Daily probability deltas by event category.
  - Calibration metrics vs. previous weeks.
  - Indicator drift (PSI, z-score shifts).
- **Alerts** – Schedule jobs (Airflow/Cron) that:
  - Email or Slack when probability change exceeds an absolute 0.2 or PSI threshold.
  - Flag confidence intervals wider than a configured limit (e.g., >0.4 span).
  - Report data ingestion failures (missing indicators, API errors).
- **Reporting** – Produce weekly Markdown/PDF summaries stored in
  `reports/probability_shift_<YYYY-MM-DD>.md` including top drivers and any manual
  overrides.

## Deployment readiness

- **Documentation** – Maintain `docs/DEPLOYMENT.md` (this file) plus a runbook in
  `docs/runbooks/current_events.md` covering restart procedures, API keys, and
  troubleshooting steps.
- **Operational checklists**:
  - CI workflow that lints inference code, runs smoke predictions on fixtures, and
    validates indicator freshness.
  - Blue/green deployment playbook for the REST service (container image tags,
    rollback steps, health-check endpoints).
  - Escalation matrix with primary/secondary contacts and vendor support channels.
- **Versioning** – Tag releases with semantic versions (e.g., `deploy-v1.2.0`) and
  archive the associated model artifacts, configs, and calibration reports under
  `deployments/<version>/` for auditability.

Adhering to this playbook ensures current-event signals are integrated with the
modeling pipeline, predictions remain calibrated, and operations teams have clear
procedures for monitoring and escalation.
