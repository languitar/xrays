from typing import Dict

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from natsort import natsorted
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def hotspots_figure(data: pd.DataFrame, cutoff: int = 10) -> go.Figure:
    data = data.groupby("file", as_index=False).agg(
        {"commit": "count", "lines_code": "last", "indentation": "last"}
    )
    data = data.rename(columns={"commit": "revisions"})

    data["urgency"] = (data["indentation"] / data["indentation"].max()) * (
        data["revisions"] / data["revisions"].max()
    ).pow(1.0 / 2)

    data = data[data["revisions"] >= cutoff]

    fig = px.scatter(
        data,
        x="revisions",
        y="lines_code",
        size="indentation",
        hover_name="file",
        color="urgency",
        color_continuous_scale="dense",
    )
    fig.update_traces(marker={"line": {"width": 1, "color": "gray"}})
    fig.update_layout(height=1000)

    return fig


def compute_correlations(data: pd.DataFrame, cutoff: int = 10) -> pd.DataFrame:
    correlation = (
        data.merge(data, on="commit", how="inner")
        .groupby(["file_x", "file_y"], as_index=False)["commit"]
        .count()
        .sort_values(["file_x", "file_y"])
    )
    # kill diagonal
    correlation.loc[correlation["file_x"] == correlation["file_y"], "commit"] = np.nan
    # cutoff
    correlation = correlation[correlation["commit"] >= cutoff]

    return correlation[["file_x", "file_y", "commit"]].rename(
        columns={"commit": "commits"}
    )


def correlation_figure(data: pd.DataFrame, cutoff: int = 10) -> go.Figure:
    correlation = compute_correlations(data, cutoff)

    sort_order = natsorted(correlation["file_x"].unique())

    fig = px.density_heatmap(
        correlation,
        x="file_x",
        y="file_y",
        z="commits",
        color_continuous_scale="dense",
        category_orders={"file_x": sort_order, "file_y": sort_order},
    )
    fig.update_layout(xaxis_title="file", yaxis_title="file", height=1000)
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    return fig


def correlation_table_data(data: pd.DataFrame, cutoff: int = 10) -> pd.DataFrame:
    data = compute_correlations(data, cutoff)
    data["duplicates"] = data[["file_x", "file_y"]].apply(
        lambda x: "-".join(sorted(x)), axis=1
    )
    data = data.drop_duplicates(subset=["duplicates"])
    del data["duplicates"]
    return data.sort_values(["commits", "file_x", "file_y"], ascending=False)


def filter_data(data: pd.DataFrame, file_regex: str) -> pd.DataFrame:
    return data[data["file"].str.contains(file_regex)]


file_regex_id = "file_regex"
hotspots_figure_id = "file_hotspots"
revision_cutoff_id = "correlation_cutoff"
correlation_figure_id = "correlations"
correlation_table_id = "correlation_table"
tab_file_hotspots_id = "file-hotspots"
tab_file_change_coupling_id = "file-change-coupling"
tab_id = "tabs"
tab_content_id = "tab-content"


def common_filters():
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Label(children="File path RegEx", htmlFor=file_regex_id),
                    dcc.Input(
                        id=file_regex_id, type="text", value=r"\.py$", debounce=True
                    ),
                ]
            ),
            html.Div(
                children=[
                    html.Label(children="Revision cutoff", htmlFor=revision_cutoff_id),
                    dcc.Slider(
                        id=revision_cutoff_id,
                        min=10,
                        max=50,
                        value=20,
                        step=1,
                        marks={i: str(i) for i in range(51)},
                    ),
                ]
            ),
        ]
    )


def create_app(data: pd.DataFrame) -> dash.Dash:
    app = dash.Dash(
        __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    )

    corr_table_data = correlation_table_data(data, 10)

    tab_render_mapping = {
        tab_file_hotspots_id: lambda data: dcc.Graph(
            id=hotspots_figure_id, figure=hotspots_figure(data)
        ),
        tab_file_change_coupling_id: lambda data: html.Div(
            children=[
                dcc.Graph(id=correlation_figure_id, figure=correlation_figure(data)),
                dash_table.DataTable(
                    id=correlation_table_id,
                    data=corr_table_data.to_dict("records"),
                    columns=[{"id": c, "name": c} for c in corr_table_data.columns],
                    page_size=20,
                ),
            ],
        ),
    }

    app.layout = html.Div(
        children=[
            html.H1(children="Xrays"),
            common_filters(),
            dcc.Tabs(
                id=tab_id,
                value=tab_file_hotspots_id,
                children=[
                    dcc.Tab(label="File Hotspots", value=tab_file_hotspots_id),
                    dcc.Tab(
                        label="File Change Coupling", value=tab_file_change_coupling_id
                    ),
                ],
            ),
            html.Div(id=tab_content_id),
        ]
    )
    app.validation_layout = html.Div(
        app.layout.children + [f(data) for f in tab_render_mapping.values()]
    )

    @app.callback(
        dash.dependencies.Output(tab_content_id, "children"),
        [dash.dependencies.Input(tab_id, "value")],
    )
    def render_tabs(tab):
        return tab_render_mapping[tab](data)

    @app.callback(
        dash.dependencies.Output(hotspots_figure_id, "figure"),
        [
            dash.dependencies.Input(file_regex_id, "value"),
            dash.dependencies.Input(revision_cutoff_id, "value"),
        ],
    )
    def update_hotspot_figure(file_regex: str, cutoff: int) -> go.Figure:
        return hotspots_figure(filter_data(data, file_regex), cutoff)

    @app.callback(
        dash.dependencies.Output(correlation_figure_id, "figure"),
        [
            dash.dependencies.Input(revision_cutoff_id, "value"),
            dash.dependencies.Input(file_regex_id, "value"),
        ],
    )
    def update_correlation_figure(cutoff: int, file_regex: str) -> go.Figure:
        return correlation_figure(filter_data(data, file_regex), cutoff)

    @app.callback(
        dash.dependencies.Output(correlation_table_id, "data"),
        [
            dash.dependencies.Input(revision_cutoff_id, "value"),
            dash.dependencies.Input(file_regex_id, "value"),
        ],
    )
    def update_correlation_table(cutoff: int, file_regex: str) -> Dict:
        return correlation_table_data(filter_data(data, file_regex), cutoff).to_dict(
            "records"
        )

    return app
