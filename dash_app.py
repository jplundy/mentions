"""Simple Dash web application for exploring transcript inventory data."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html
import plotly.express as px
import plotly.graph_objects as go
import yaml

LOGGER = logging.getLogger(__name__)
DATA_DIR = Path("data")
EXPERIMENTS_DIR = Path("experiments")


@dataclass
class InventoryPayload:
    """Serializable payload representing a loaded inventory dataset."""

    records: List[Dict[str, object]]
    columns: List[str]
    event_types: List[str]
    date_column: Optional[str]
    date_min: Optional[str]
    date_max: Optional[str]
    target_terms: List[str]
    dataset_label: str
    project_key: Optional[str]
    baseline_metrics: List["BaselineMetric"] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "records": self.records,
            "columns": self.columns,
            "event_types": self.event_types,
            "date_column": self.date_column,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "target_terms": self.target_terms,
            "dataset_label": self.dataset_label,
            "project_key": self.project_key,
            "baseline_metrics": [metric.as_dict() for metric in self.baseline_metrics],
        }


@dataclass
class BaselineMetric:
    """Aggregated performance metrics for a tracked target phrase."""

    phrase: str
    experiment: str
    target_column: Optional[str]
    dataset_path: Optional[Path]
    dataset_version: Optional[str]
    generated_at: Optional[str]
    num_rows: Optional[int]
    log_loss: Optional[float]
    brier: Optional[float]
    accuracy: Optional[float]
    average_precision: Optional[float]
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

    @property
    def health_status(self) -> str:
        """Return 'degenerate', 'warning', or 'healthy' based on aggregate metrics."""
        if self.f1 is not None and self.f1 == 0.0:
            return "degenerate"
        if self.roc_auc is None:
            return "warning"
        return "healthy"

    def as_dict(self) -> Dict[str, object]:
        return {
            "phrase": self.phrase,
            "experiment": self.experiment,
            "target_column": self.target_column,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "dataset_version": self.dataset_version,
            "generated_at": self.generated_at,
            "num_rows": self.num_rows,
            "log_loss": self.log_loss,
            "brier": self.brier,
            "accuracy": self.accuracy,
            "average_precision": self.average_precision,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "health_status": self.health_status,
        }


def discover_inventory_files(data_dir: Path = DATA_DIR) -> List[Path]:
    """Return a sorted list of available inventory CSV files."""

    if not data_dir.exists():
        return []
    candidates = list(data_dir.glob("**/*inventory*.csv"))
    return sorted(p for p in candidates if p.is_file())


def infer_project_key(path: Path) -> Optional[str]:
    """Infer a project key (e.g., `nflx`) from an inventory filename."""

    stem = path.stem
    for suffix in ("_inventory", "-inventory"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or None


def load_target_terms(project_key: Optional[str], data_dir: Path = DATA_DIR) -> List[str]:
    """Return sorted target terms from derivative manifests, if available."""

    if not project_key:
        return []
    derivatives_dir = data_dir / "derivatives" / project_key
    if not derivatives_dir.exists():
        return []

    manifests = sorted(derivatives_dir.glob("segments_*.json"))
    for manifest_path in manifests:
        try:
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                payload = json.load(manifest_file)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load manifest %s: %s", manifest_path, exc)
            continue
        target_columns = payload.get("target_columns")
        if isinstance(target_columns, dict) and target_columns:
            return sorted(target_columns.keys())
    return []


def load_baseline_metrics(
    project_key: Optional[str],
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> List[BaselineMetric]:
    """Load aggregated baseline metrics for tracked phrases in a project."""

    if not project_key or not experiments_dir.exists():
        return []

    pattern = f"{project_key}_baseline*"
    candidates = sorted(p for p in experiments_dir.glob(pattern) if p.is_dir())
    metrics: List[BaselineMetric] = []
    for run_dir in candidates:
        metric = _load_baseline_metric(run_dir)
        if metric:
            metrics.append(metric)

    metrics.sort(key=lambda item: (item.phrase.lower(), item.experiment))
    return metrics


def _load_baseline_metric(run_dir: Path) -> Optional[BaselineMetric]:
    """Return baseline metrics parsed from a single experiment directory."""

    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "aggregate_metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        return None

    try:
        config = yaml.safe_load(config_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to parse baseline config %s: %s", config_path, exc)
        return None

    target_column = config.get("target_column") if isinstance(config, dict) else None
    dataset_path_value = config.get("dataset_path") if isinstance(config, dict) else None
    dataset_path = Path(dataset_path_value) if dataset_path_value else None

    manifest_data: Optional[Dict[str, object]] = None
    manifest_path = _resolve_manifest_path(dataset_path) if dataset_path else None
    if manifest_path and manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                manifest_data = json.load(manifest_file)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load manifest %s: %s", manifest_path, exc)

    phrase = None
    dataset_version = None
    generated_at = None
    num_rows = None
    if isinstance(manifest_data, dict):
        dataset_version = manifest_data.get("version")
        generated_at = manifest_data.get("generated_at")
        num_rows = manifest_data.get("num_rows")
        target_map = manifest_data.get("target_columns") or {}
        if isinstance(target_map, dict) and target_column:
            phrase = next(
                (label for label, column in target_map.items() if column == target_column),
                None,
            )

    clean_phrase = _humanize_target_column(target_column, phrase)

    try:
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            raw_metrics = json.load(metrics_file)
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to load metrics %s: %s", metrics_path, exc)
        return None

    return BaselineMetric(
        phrase=clean_phrase,
        experiment=run_dir.name,
        target_column=target_column,
        dataset_path=dataset_path,
        dataset_version=dataset_version,
        generated_at=generated_at,
        num_rows=num_rows if isinstance(num_rows, int) else None,
        log_loss=_clean_metric(raw_metrics.get("log_loss")),
        brier=_clean_metric(raw_metrics.get("brier")),
        accuracy=_clean_metric(raw_metrics.get("accuracy")),
        average_precision=_clean_metric(raw_metrics.get("average_precision")),
        f1=_clean_metric(raw_metrics.get("f1")),
        roc_auc=_clean_metric(raw_metrics.get("roc_auc")),
        precision=_clean_metric(raw_metrics.get("precision")),
        recall=_clean_metric(raw_metrics.get("recall")),
    )


def _clean_metric(value: object) -> Optional[float]:
    """Return a float metric or ``None`` if the value is missing/invalid."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _humanize_target_column(
    target_column: Optional[str], fallback: Optional[str]
) -> str:
    """Convert a target column slug into a human-readable phrase."""

    if fallback:
        return fallback
    if not target_column:
        return "Tracked phrase"
    label = target_column
    if label.startswith("target__"):
        label = label[len("target__") :]
    return label.replace("_", " ")


def load_fold_metrics(
    experiment_name: str,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> List[Dict[str, object]]:
    """Load per-fold metric dicts from an experiment directory."""

    run_dir = experiments_dir / experiment_name
    if not run_dir.is_dir():
        return []
    fold_files = sorted(run_dir.glob("fold_*_metrics.json"))
    folds: List[Dict[str, object]] = []
    for fold_path in fold_files:
        try:
            data = json.loads(fold_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                folds.append(data)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Failed to load fold metrics %s: %s", fold_path, exc)
    return folds


def _resolve_manifest_path(dataset_path: Path) -> Optional[Path]:
    """Infer the JSON manifest associated with a dataset path."""

    if not dataset_path:
        return None

    candidates = []
    try:
        candidates.append(dataset_path.with_suffix(".json"))
    except ValueError:  # pragma: no cover - unlikely but defensive
        pass

    parent = dataset_path.parent
    stem = dataset_path.stem
    if parent.exists():
        candidates.extend(parent.glob(f"{stem}.json"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _format_percent(value: Optional[float]) -> str:
    """Return a human-readable percentage or em dash for missing values."""

    if value is None:
        return "—"
    return f"{value:.1%}"


def _format_score(value: Optional[float]) -> str:
    """Return a compact numeric string for scalar metrics."""

    if value is None:
        return "—"
    return f"{value:.3f}"


def _empty_baseline_figure(message: str) -> go.Figure:
    """Return a placeholder figure with a centered annotation."""

    fig = go.Figure()
    fig.update_layout(
        title="Baseline accuracy by tracked phrase",
        template="plotly_white",
        xaxis_title="Tracked phrase",
        yaxis_title="Accuracy",
        yaxis_tickformat=".0%",
    )
    fig.add_annotation(
        text=message,
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        font={"color": "#666"},
    )
    return fig


def _choose_date_column(columns: Iterable[str]) -> Optional[str]:
    """Return the best candidate column to interpret as a date."""

    date_candidates = [
        "event_date",
        "date",
        "timestamp",
        "datetime",
    ]
    normalized = {col.lower(): col for col in columns}
    for candidate in date_candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_inventory_payload(path: Path) -> InventoryPayload:
    """Load an inventory CSV and prepare a serializable payload for the app."""

    dataset_label = path.name
    project_key = infer_project_key(path)

    try:
        df = pd.read_csv(path)
    except OSError as exc:  # pragma: no cover - defensive
        LOGGER.error("Unable to read %s: %s", path, exc)
        return InventoryPayload([], [], [], None, None, None, [], dataset_label, project_key)

    date_column = _choose_date_column(df.columns)
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    event_types: List[str] = []
    if "event_type" in df.columns:
        series = df["event_type"].dropna().astype(str)
        event_types = sorted(series.unique())

    date_min = date_max = None
    if date_column:
        date_values = df[date_column].dropna()
        if not date_values.empty:
            date_min = date_values.min().date().isoformat()
            date_max = date_values.max().date().isoformat()
        df[date_column] = df[date_column].dt.strftime("%Y-%m-%d")

    columns = list(df.columns)
    records = df.to_dict("records")

    target_terms = load_target_terms(project_key)
    baseline_metrics = load_baseline_metrics(project_key)

    return InventoryPayload(
        records=records,
        columns=columns,
        event_types=event_types,
        date_column=date_column,
        date_min=date_min,
        date_max=date_max,
        target_terms=target_terms,
        dataset_label=dataset_label,
        project_key=project_key,
        baseline_metrics=baseline_metrics,
    )


def build_dash_app() -> Dash:
    """Instantiate the Dash application and register callbacks."""

    inventory_files = discover_inventory_files()
    options = [
        {"label": path.name, "value": str(path)}
        for path in inventory_files
    ]
    default_value = options[0]["value"] if options else None

    app = Dash(__name__, title="Transcript Explorer")
    app.layout = html.Div(
        className="app-container",
        children=[
            html.H1("Transcript Data Explorer"),
            html.P(
                "Interactively explore transcript inventories, filter by event type, "
                "and review tracked keyword targets."
            ),
            dcc.Dropdown(
                id="inventory-selector",
                options=options,
                value=default_value,
                placeholder="Select an inventory CSV",
                clearable=False,
            ),
            dcc.Store(id="inventory-store"),
            html.Div(id="dataset-summary", className="summary-card"),
            html.Div(
                className="filter-row",
                children=[
                    html.Div(
                        children=[
                            html.Label("Event type"),
                            dcc.Dropdown(
                                id="event-type-filter",
                                options=[],
                                value=[],
                                multi=True,
                                placeholder="Filter by event type",
                                clearable=True,
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Event date range"),
                            dcc.DatePickerRange(
                                id="date-range",
                                minimum_nights=0,
                                start_date_placeholder_text="Start",
                                end_date_placeholder_text="End",
                            ),
                        ]
                    ),
                ],
            ),
            dcc.Graph(id="inventory-timeline", config={"displaylogo": False}),
            dash_table.DataTable(
                id="inventory-table",
                data=[],
                columns=[],
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "0.5rem"},
                style_header={"fontWeight": "bold"},
            ),
            html.Div(id="target-summary", className="summary-card"),
            html.Div(
                className="baseline-card",
                children=[
                    html.H2("Baseline tracked phrase performance"),
                    html.P(
                        "Metrics loaded from baseline experiment outputs associated with this "
                        "inventory. Values summarize how historical models performed for each "
                        "tracked phrase."
                    ),
                    html.Div(id="model-health-alerts"),
                    html.Div(
                        className="metric-selector-row",
                        children=[
                            html.Label("Primary metric"),
                            dcc.Dropdown(
                                id="metric-selector",
                                options=[
                                    {"label": "F1 Score", "value": "f1"},
                                    {"label": "Average Precision (AP)", "value": "average_precision"},
                                    {"label": "ROC AUC", "value": "roc_auc"},
                                    {"label": "Accuracy", "value": "accuracy"},
                                    {"label": "Log Loss (lower = better)", "value": "log_loss"},
                                    {"label": "Brier Score (lower = better)", "value": "brier"},
                                ],
                                value="f1",
                                clearable=False,
                                style={"width": "320px"},
                            ),
                        ],
                    ),
                    dcc.Graph(
                        id="baseline-performance-graph",
                        config={"displaylogo": False},
                    ),
                    dash_table.DataTable(
                        id="baseline-metrics-table",
                        data=[],
                        columns=[],
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "0.5rem"},
                        style_header={"fontWeight": "bold"},
                    ),
                    html.Div(
                        className="fold-card",
                        children=[
                            html.H3("Fold-level metric distribution"),
                            html.P(
                                "Per-fold metrics reveal variance and stability across "
                                "validation splits. High variance or NaN values indicate "
                                "data quality or class-imbalance issues."
                            ),
                            html.Div(
                                children=[
                                    html.Label("Experiment"),
                                    dcc.Dropdown(
                                        id="fold-experiment-selector",
                                        options=[],
                                        value=None,
                                        clearable=True,
                                        placeholder="Select an experiment to inspect folds",
                                        style={"width": "400px"},
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id="fold-distribution-graph",
                                config={"displaylogo": False},
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("inventory-store", "data"),
        Output("dataset-summary", "children"),
        Input("inventory-selector", "value"),
        prevent_initial_call=False,
    )
    def update_inventory_store(path_value: Optional[str]):
        if not path_value:
            return {}, html.P("No inventory selected.")
        payload = load_inventory_payload(Path(path_value))
        summary = build_dataset_summary(payload)
        return payload.as_dict(), summary

    @app.callback(
        Output("event-type-filter", "options"),
        Output("event-type-filter", "value"),
        Output("date-range", "min_date_allowed"),
        Output("date-range", "max_date_allowed"),
        Output("date-range", "start_date"),
        Output("date-range", "end_date"),
        Output("target-summary", "children"),
        Input("inventory-store", "data"),
        prevent_initial_call=False,
    )
    def update_filter_controls(store_data: Optional[Dict[str, object]]):
        if not store_data:
            return [], [], None, None, None, None, html.P("Select a dataset to view target keywords.")

        event_types = [
            {"label": event_type, "value": event_type}
            for event_type in store_data.get("event_types", [])
        ]
        target_terms = store_data.get("target_terms", [])
        if target_terms:
            target_children = html.Div(
                [
                    html.H3("Tracked keyword targets"),
                    html.P(
                        "Keyword-based boolean columns generated by the pipeline. "
                        "Use them to focus modeling or quality checks."
                    ),
                    html.Div(
                        className="tag-grid",
                        children=[html.Span(term, className="tag") for term in target_terms],
                    ),
                ]
            )
        else:
            target_children = html.Div(
                [
                    html.H3("Tracked keyword targets"),
                    html.P("No keyword metadata found for this dataset."),
                ]
            )

        return (
            event_types,
            [],
            store_data.get("date_min"),
            store_data.get("date_max"),
            store_data.get("date_min"),
            store_data.get("date_max"),
            target_children,
        )

    # Metric labels used in chart titles and axis labels.
    _METRIC_LABELS: Dict[str, str] = {
        "f1": "F1 Score",
        "average_precision": "Average Precision",
        "roc_auc": "ROC AUC",
        "accuracy": "Accuracy",
        "log_loss": "Log Loss",
        "brier": "Brier Score",
    }
    _PCT_METRICS = {"f1", "average_precision", "roc_auc", "accuracy"}

    @app.callback(
        Output("model-health-alerts", "children"),
        Input("inventory-store", "data"),
        prevent_initial_call=False,
    )
    def update_health_alerts(store_data: Optional[Dict[str, object]]):
        if not store_data:
            return []
        baseline_metrics = store_data.get("baseline_metrics") or []
        degenerate = [m for m in baseline_metrics if m.get("health_status") == "degenerate"]
        warning = [m for m in baseline_metrics if m.get("health_status") == "warning"]
        alerts = []
        if degenerate:
            names = ", ".join(sorted({m.get("experiment", "?") for m in degenerate}))
            alerts.append(
                html.Div(
                    [
                        html.Strong("Degenerate model detected: "),
                        html.Span(
                            f"F1=0 in {names}. The model predicts only the majority class. "
                            "Check class imbalance and decision threshold."
                        ),
                    ],
                    style={
                        "background": "#fee2e2",
                        "border": "1px solid #f87171",
                        "borderRadius": "4px",
                        "padding": "0.75rem 1rem",
                        "marginBottom": "0.5rem",
                        "color": "#7f1d1d",
                    },
                )
            )
        if warning:
            names = ", ".join(sorted({m.get("experiment", "?") for m in warning}))
            alerts.append(
                html.Div(
                    [
                        html.Strong("ROC AUC undefined: "),
                        html.Span(
                            f"NaN ROC AUC in {names}. Some validation folds contained "
                            "only one class. Consider increasing minimum_train_events."
                        ),
                    ],
                    style={
                        "background": "#fef9c3",
                        "border": "1px solid #fbbf24",
                        "borderRadius": "4px",
                        "padding": "0.75rem 1rem",
                        "marginBottom": "0.5rem",
                        "color": "#713f12",
                    },
                )
            )
        return alerts

    @app.callback(
        Output("baseline-performance-graph", "figure"),
        Output("baseline-metrics-table", "data"),
        Output("baseline-metrics-table", "columns"),
        Input("inventory-store", "data"),
        Input("metric-selector", "value"),
        prevent_initial_call=False,
    )
    def update_baseline_section(
        store_data: Optional[Dict[str, object]],
        selected_metric: Optional[str],
    ):
        metric_key = selected_metric or "f1"
        metric_label = _METRIC_LABELS.get(metric_key, metric_key)

        if not store_data:
            return _empty_baseline_figure("Select a dataset to load baseline experiments."), [], []

        baseline_metrics = store_data.get("baseline_metrics") or []
        if not baseline_metrics:
            return _empty_baseline_figure(
                "No baseline experiments discovered for this dataset."
            ), [], []

        baseline_df = pd.DataFrame(baseline_metrics)
        all_numeric = list(_METRIC_LABELS.keys())
        for column in all_numeric:
            if column in baseline_df.columns:
                baseline_df[column] = pd.to_numeric(baseline_df[column], errors="coerce")
        metric_df = baseline_df.dropna(subset=[metric_key]).copy() if metric_key in baseline_df.columns else pd.DataFrame()
        metric_df.sort_values(["phrase", "experiment"], inplace=True)

        if metric_df.empty:
            figure = _empty_baseline_figure(
                f"No experiments reported {metric_label}."
            )
        else:
            use_pct = metric_key in _PCT_METRICS
            figure = px.bar(
                metric_df,
                x="phrase",
                y=metric_key,
                color="experiment",
                title=f"{metric_label} by tracked phrase",
                labels={
                    "phrase": "Tracked phrase",
                    metric_key: metric_label,
                    "experiment": "Experiment",
                },
            )
            figure.update_layout(
                template="plotly_white",
                yaxis_tickformat=".0%" if use_pct else ".3f",
                legend_title_text="Experiment",
            )
            hover_cols = ["experiment", "f1", "average_precision", "accuracy", "log_loss"]
            available = [c for c in hover_cols if c in metric_df.columns]
            customdata = metric_df[available].to_numpy()
            fmt = ":.2%" if use_pct else ":.3f"
            figure.update_traces(
                customdata=customdata,
                hovertemplate=(
                    f"<b>%{{x}}</b><br>"
                    f"Experiment=%{{customdata[0]}}<br>"
                    f"{metric_label}=%{{y{fmt}}}<br>"
                    f"F1=%{{customdata[1]:.3f}}<br>"
                    f"Avg Precision=%{{customdata[2]:.3f}}<br>"
                    f"Accuracy=%{{customdata[3]:.2%}}<br>"
                    f"Log Loss=%{{customdata[4]:.3f}}<extra></extra>"
                ),
            )

        table_rows: List[Dict[str, object]] = []
        for metric in baseline_metrics:
            status = metric.get("health_status", "")
            status_icon = {"degenerate": "DEGENERATE", "warning": "WARNING", "healthy": "OK"}.get(status, "—")
            row = {
                "health": status_icon,
                "phrase": metric.get("phrase"),
                "experiment": metric.get("experiment"),
                "dataset_version": metric.get("dataset_version") or "—",
                "num_rows": metric.get("num_rows") if metric.get("num_rows") is not None else "—",
                "f1": _format_score(metric.get("f1")),
                "precision": _format_score(metric.get("precision")),
                "recall": _format_score(metric.get("recall")),
                "average_precision": _format_percent(metric.get("average_precision")),
                "roc_auc": _format_score(metric.get("roc_auc")),
                "accuracy": _format_percent(metric.get("accuracy")),
                "log_loss": _format_score(metric.get("log_loss")),
                "brier": _format_score(metric.get("brier")),
            }
            table_rows.append(row)

        columns = [
            {"name": "Health", "id": "health"},
            {"name": "Tracked phrase", "id": "phrase"},
            {"name": "Experiment", "id": "experiment"},
            {"name": "Dataset version", "id": "dataset_version"},
            {"name": "Rows", "id": "num_rows"},
            {"name": "F1", "id": "f1"},
            {"name": "Precision", "id": "precision"},
            {"name": "Recall", "id": "recall"},
            {"name": "Avg Precision", "id": "average_precision"},
            {"name": "ROC AUC", "id": "roc_auc"},
            {"name": "Accuracy", "id": "accuracy"},
            {"name": "Log Loss", "id": "log_loss"},
            {"name": "Brier", "id": "brier"},
        ]

        return figure, table_rows, columns

    @app.callback(
        Output("fold-experiment-selector", "options"),
        Output("fold-experiment-selector", "value"),
        Input("inventory-store", "data"),
        prevent_initial_call=False,
    )
    def update_fold_experiment_selector(store_data: Optional[Dict[str, object]]):
        if not store_data:
            return [], None
        baseline_metrics = store_data.get("baseline_metrics") or []
        seen: Dict[str, str] = {}
        for m in baseline_metrics:
            exp = m.get("experiment")
            phrase = m.get("phrase", exp)
            if exp and exp not in seen:
                seen[exp] = f"{exp} ({phrase})"
        options = [{"label": label, "value": exp} for exp, label in seen.items()]
        default = options[0]["value"] if options else None
        return options, default

    @app.callback(
        Output("fold-distribution-graph", "figure"),
        Input("fold-experiment-selector", "value"),
        Input("metric-selector", "value"),
        prevent_initial_call=False,
    )
    def update_fold_distribution(
        experiment_name: Optional[str],
        selected_metric: Optional[str],
    ):
        metric_key = selected_metric or "f1"
        metric_label = _METRIC_LABELS.get(metric_key, metric_key)
        use_pct = metric_key in _PCT_METRICS

        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Select an experiment to view fold-level metrics",
            template="plotly_white",
        )

        if not experiment_name:
            return empty_fig

        fold_data = load_fold_metrics(experiment_name)
        if not fold_data:
            empty_fig.update_layout(
                title=f"No fold metrics found for {experiment_name}"
            )
            return empty_fig

        rows = []
        for idx, fold in enumerate(fold_data):
            for key, val in fold.items():
                if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                    rows.append({"fold": idx, "metric": key, "value": val})

        if not rows:
            empty_fig.update_layout(title=f"All fold metrics are NaN for {experiment_name}")
            return empty_fig

        fold_df = pd.DataFrame(rows)
        metric_subset = fold_df[fold_df["metric"] == metric_key]

        if metric_subset.empty:
            empty_fig.update_layout(
                title=f"{metric_label} not available for {experiment_name}"
            )
            return empty_fig

        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y=metric_subset["value"].tolist(),
                name=metric_label,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                marker_color="#4f86c6",
            )
        )
        fig.update_layout(
            title=f"{metric_label} across {len(metric_subset)} folds — {experiment_name}",
            template="plotly_white",
            yaxis_title=metric_label,
            yaxis_tickformat=".0%" if use_pct else ".3f",
            showlegend=False,
        )
        return fig

    @app.callback(
        Output("inventory-timeline", "figure"),
        Output("inventory-table", "data"),
        Output("inventory-table", "columns"),
        Input("inventory-store", "data"),
        Input("event-type-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        prevent_initial_call=False,
    )
    def update_visuals(
        store_data: Optional[Dict[str, object]],
        event_type_values: Optional[List[str]],
        start_date: Optional[str],
        end_date: Optional[str],
    ):
        if not store_data or not store_data.get("records"):
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No inventory data available",
                template="plotly_white",
                xaxis_title="Event date",
                yaxis_title="Event count",
            )
            return empty_fig, [], []

        df = pd.DataFrame(store_data["records"])
        date_column = store_data.get("date_column")

        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df = df.dropna(subset=[date_column])

        if event_type_values:
            df = df[df["event_type"].isin(event_type_values)]

        if date_column:
            if start_date:
                df = df[df[date_column] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df[date_column] <= pd.to_datetime(end_date)]

        if date_column:
            display_df = df.copy()
            display_df[date_column] = display_df[date_column].dt.strftime("%Y-%m-%d")
        else:
            display_df = df

        columns = [
            {"name": col.replace("_", " ").title(), "id": col}
            for col in display_df.columns
        ]
        data = display_df.to_dict("records")

        if date_column and not df.empty:
            counts = (
                df.groupby(date_column)
                .size()
                .reset_index(name="count")
                .sort_values(date_column)
            )
            fig = px.bar(
                counts,
                x=date_column,
                y="count",
                labels={"count": "Events", date_column: "Event date"},
                title="Events over time",
            )
            fig.update_layout(template="plotly_white")
        else:
            fig = go.Figure()
            fig.update_layout(
                title="Events over time",
                template="plotly_white",
                xaxis_title="Event date",
                yaxis_title="Event count",
            )

        return fig, data, columns

    return app


def build_dataset_summary(payload: InventoryPayload) -> html.Div:
    """Construct a short natural-language summary for the dataset card."""

    if not payload.records:
        return html.Div(
            [
                html.H3(payload.dataset_label),
                html.P("No rows were found in the selected inventory."),
            ],
            className="summary-card",
        )

    count = len(payload.records)
    summary_parts = [
        html.H3(payload.dataset_label),
        html.P(f"{count} events loaded."),
    ]
    if payload.date_min and payload.date_max:
        summary_parts.append(
            html.P(f"Coverage: {payload.date_min} → {payload.date_max}.")
        )
    if payload.project_key:
        summary_parts.append(html.P(f"Project key: {payload.project_key}"))

    return html.Div(summary_parts, className="summary-card")


def main() -> None:
    app = build_dash_app()
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
