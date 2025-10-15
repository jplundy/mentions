"""Simple Dash web application for exploring transcript inventory data."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html
import plotly.express as px
import plotly.graph_objects as go

LOGGER = logging.getLogger(__name__)
DATA_DIR = Path("data")


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
