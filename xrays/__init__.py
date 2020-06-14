import itertools
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import Dict, List, Pattern, Tuple

import click
import click_pathlib
import pandas as pd
import plotly.express as px
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


@hotspots.command()
@click.argument("data_dir", type=click_pathlib.Path(exists=True))
@click.argument("viz_dir", type=click_pathlib.Path())
def visualize(data_dir: Path, viz_dir: Path) -> None:
    viz_dir.mkdir(exist_ok=True)

    data = pd.read_parquet(hotspot_data_file(data_dir))

    data = data.groupby("file", as_index=False).agg(
        {"commit": "count", "lines_code": "first", "indentation": "first"}
    )
    data = data.rename(columns={"commit": "revisions"})

    data["urgency"] = (data["indentation"] / data["indentation"].max()) * (
        data["revisions"] / data["revisions"].max()
    ).pow(1.0 / 2)

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

    out_file = viz_dir / "hotspots.html"
    fig.write_html(str(out_file))

    fig.show()


if __name__ == "__main__":
    hotspots()
