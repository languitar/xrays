import itertools
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import List, Pattern, Tuple

import click
import click_pathlib
import dash
import dash_core_components as dcc
import dash_html_components as html
from natsort import natsorted
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm


def hotspot_data_file(out_path: Path) -> Path:
    return out_path / "hotspots.parquet"


def file_revision_information(git_root: Path, file: Path) -> pd.DataFrame:
    log_out = subprocess.run(
        [
            "git",
            "-C",
            str(git_root),
            "log",
            "--format=format:%H %aI %cI",
            "--follow",
            "--",
            file,
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()

    commit = []
    author_date = []
    commit_date = []
    for line in log_out:
        parts = line.split(" ")
        commit.append(parts[0])
        author_date.append(parts[1])
        commit_date.append(parts[2])

    return pd.DataFrame(
        {
            "file": str(file),
            "commit": commit,
            "author_date": pd.to_datetime(author_date, utc=True),
            "commit_date": pd.to_datetime(commit_date, utc=True),
        }
    )


def count_lines(file: Path) -> Tuple[int, int]:
    cloc_out = subprocess.run(
        ["cloc", "--json", file], check=True, stdout=subprocess.PIPE
    ).stdout
    if cloc_out.strip():
        cloc_data = json.loads(cloc_out)
        return cloc_data["SUM"]["code"], cloc_data["SUM"]["comment"]
    else:
        return 0, 0


STRIPPED_EXT = "stripped"


def count_indentations(file: Path) -> int:
    with TemporaryDirectory() as work_dir:
        subprocess.run(
            ["cloc", f"--strip-comments={STRIPPED_EXT}", file],
            check=True,
            cwd=work_dir,
            stdout=subprocess.PIPE,
        )

        try:
            stripped_content = (
                Path(work_dir) / f"{file.name}.{STRIPPED_EXT}"
            ).read_text()
            # TODO hard-coded assumption
            stripped_content = stripped_content.replace("\t", 4 * " ")
            lines = stripped_content.splitlines()
            counts = [
                sum(1 for _ in itertools.takewhile(str.isspace, line)) for line in lines
            ]
            return sum(counts)
        except FileNotFoundError:
            # cloc skips empty files
            return 0


def relevant_files_in_git_root(git_root: Path, pattern: Pattern) -> List[Path]:
    files_out = subprocess.run(
        ["git", "-C", str(git_root), "ls-files"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    files = files_out.splitlines()
    return [Path(f) for f in files if pattern.fullmatch(f)]


def acquire_file_base_data(git_root: Path, file: Path) -> pd.DataFrame:
    data = file_revision_information(git_root, file)
    data["indentation"] = count_indentations(git_root / file)
    lines_code, line_comment = count_lines(git_root / file)
    data["lines_code"] = lines_code
    data["lines_comment"] = line_comment

    return data


def acquire_base_data(git_root: Path, pattern: Pattern) -> pd.DataFrame:
    files = relevant_files_in_git_root(git_root, pattern)
    print(f"Analyzing {len(files)} files")

    per_file_data = []
    for file in tqdm(files):
        per_file_data.append(acquire_file_base_data(git_root, file))

    return pd.concat(per_file_data)


@click.group()
def hotspots() -> None:
    pass


@hotspots.command()
@click.option(
    "--file-pattern",
    default=".*",
    help="Only include files in the repository matching this regular expression",
)
@click.argument("git_root", type=click_pathlib.Path())
@click.argument("data_dir", type=click_pathlib.Path())
def compute(git_root: Path, file_pattern: str, data_dir: Path) -> None:
    data_dir.mkdir(exist_ok=True)

    data = acquire_base_data(git_root, re.compile(file_pattern))
    data.to_parquet(hotspot_data_file(data_dir))


def hotspots_figure(data: pd.DataFrame, cutoff: int = 10) -> go.Figure:
    data = data.groupby("file", as_index=False).agg(
        {"commit": "count", "lines_code": "first", "indentation": "first"}
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

    return fig


def correlation_figure(data: pd.DataFrame, cutoff: int = 10) -> go.Figure:
    correlation = (
        data.merge(data, on="commit", how="inner")
        .groupby(["file_x", "file_y"], as_index=False)["commit"]
        .count()
        .sort_values(["file_x", "file_y"])
    )
    # kill diagonal
    correlation.loc[correlation["file_x"] == correlation["file_y"], "commit"] = 0

    correlation = correlation[correlation["commit"] >= cutoff]

    sort_order = natsorted(correlation["file_x"].unique())

    fig = px.density_heatmap(
        correlation,
        x="file_x",
        y="file_y",
        z="commit",
        color_continuous_scale="dense",
        category_orders={"file_x": sort_order, "file_y": sort_order},
        labels={"commit": "commits"},
    )
    fig.update_layout(xaxis_title="file", yaxis_title="file")

    return fig


@hotspots.command()
@click.argument("data_dir", type=click_pathlib.Path(exists=True))
def visualize(data_dir: Path) -> None:
    data = pd.read_parquet(hotspot_data_file(data_dir))

    app = dash.Dash(
        __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    )

    file_regex_id = "file_regex"
    hotspots_figure_id = "file_hotspots"
    revision_cutoff_id = "correlation_cutoff"
    correlation_figure_id = "correlations"

    app.layout = html.Div(
        children=[
            html.H1(children="Xrays"),
            html.Div(children="File path RegEx"),
            dcc.Input(id=file_regex_id, type="text", value=r"\.py$", debounce=True),
            html.Div(children="Revision cutoff"),
            dcc.Slider(
                id=revision_cutoff_id,
                min=0,
                max=50,
                value=10,
                step=1,
                marks={i: str(i) for i in range(51)},
            ),
            html.H2(children="File Hotspots"),
            dcc.Graph(id=hotspots_figure_id, figure=hotspots_figure(data)),
            html.H2(children="File Correlations"),
            dcc.Graph(id=correlation_figure_id, figure=correlation_figure(data)),
        ]
    )

    def filter_data(data: pd.DataFrame, file_regex: str) -> pd.DataFrame:
        return data[data["file"].str.contains(file_regex)]

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

    app.run_server(debug=True, use_reloader=False)


if __name__ == "__main__":
    hotspots()
