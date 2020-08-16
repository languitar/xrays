from pathlib import Path
import re

import click
import click_pathlib
import pandas as pd

from xrays.analysis import acquire_base_data
from xrays.dashboard import create_app


def hotspot_data_file(out_path: Path) -> Path:
    return out_path / "hotspots.parquet"


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
def visualize(data_dir: Path) -> None:
    data = pd.read_parquet(hotspot_data_file(data_dir))
    create_app(data).run_server(debug=True)


if __name__ == "__main__":
    hotspots()
