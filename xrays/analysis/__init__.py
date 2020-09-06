import itertools
import json
import logging
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import List, Pattern, Tuple

import git
import pandas as pd
from tqdm import tqdm


_LOGGER = logging.getLogger(__name__)


def relevant_files_in_git_root(git_root: Path, pattern: Pattern) -> List[Path]:
    files_out = subprocess.run(
        ["git", "-C", str(git_root), "ls-files"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    files = files_out.splitlines()
    return [Path(f) for f in files if pattern.fullmatch(f)]


def acquire_base_data(git_root: Path, pattern: Pattern) -> pd.DataFrame:
    files = relevant_files_in_git_root(git_root, pattern)
    print(f"Analyzing {len(files)} files")

    per_file_data = []
    for file in tqdm(files):
        per_file_data.append(acquire_file_base_data(git_root, file))

    if not per_file_data:
        raise ValueError("Empty result with the given pattern.")

    return pd.concat(per_file_data)


def acquire_file_base_data(git_root: Path, file: Path) -> pd.DataFrame:
    return file_revision_information(git_root, file)


def file_revision_information(git_root: Path, file: Path) -> pd.DataFrame:
    _LOGGER.debug(
        "Acquiring per-file information for file %s in root %s", file, git_root
    )
    log_out = subprocess.run(
        [
            "git",
            "-C",
            str(git_root),
            "log",
            "--format=format:%H %aI %cI",
            "--follow",
            "--name-only",
            "--",
            file,
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    log_out.append("")

    repo = git.Repo(git_root)

    commits = []
    commit_filenames = []
    author_dates = []
    commit_dates = []
    indentations = []
    lines_code = []
    lines_comment = []
    for lines in [log_out[i : i + 3] for i in range(0, len(log_out), 3)]:
        data_line, commit_filename, _ = lines
        _LOGGER.debug("Processing output line %s", data_line)

        commit_filenames.append(commit_filename)

        commit, author_date, commit_date = data_line.split(" ")
        commits.append(commit)
        author_dates.append(author_date)
        commit_dates.append(commit_date)

        file_contents = repo.commit(commit).tree[commit_filename].data_stream.read()
        indentations.append(count_indentations(file, file_contents))

        code, comment = count_lines(file, file_contents)
        lines_code.append(code)
        lines_comment.append(comment)

    return pd.DataFrame(
        {
            "file": str(file),
            "commit_filename": commit_filenames,
            "commit": commits,
            "author_date": pd.to_datetime(author_dates, utc=True),
            "commit_date": pd.to_datetime(commit_dates, utc=True),
            "indentation": indentations,
            "lines_code": lines_code,
            "lines_comment": lines_comment,
        }
    )


def count_lines(filename: str, contents: bytes) -> Tuple[int, int]:
    with TemporaryDirectory() as work_dir:
        input_file = Path(work_dir) / filename.name
        input_file.touch()
        input_file.write_bytes(contents)

        cloc_out = subprocess.run(
            ["cloc", "--json", input_file], check=True, stdout=subprocess.PIPE
        ).stdout
        if cloc_out.strip():
            cloc_data = json.loads(cloc_out)
            return cloc_data["SUM"]["code"], cloc_data["SUM"]["comment"]
        else:
            return 0, 0


STRIPPED_EXT = "stripped"


def count_indentations(filename: str, contents: bytes) -> int:
    with TemporaryDirectory() as work_dir:
        input_file = Path(work_dir) / filename.name
        input_file.touch()
        input_file.write_bytes(contents)

        subprocess.run(
            ["cloc", f"--strip-comments={STRIPPED_EXT}", input_file],
            check=True,
            cwd=work_dir,
            stdout=subprocess.PIPE,
        )

        try:
            stripped_content = (
                Path(work_dir) / f"{input_file.name}.{STRIPPED_EXT}"
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
