from __future__ import annotations

import os
import sys
from typing import Any, List, Tuple

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = "notebooks"
NOTEBOOKS = [file for file in os.listdir(os.path.join(ROOT_DIR, NOTEBOOKS_DIR)) if file.endswith(".ipynb")]


def run_notebook(notebook_path: str) -> Tuple[Any, List[Any]]:
    full_path = os.path.join(ROOT_DIR, notebook_path)
    nb_name, _ = os.path.splitext(os.path.basename(full_path))

    with open(full_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=600, kernel_name="python3")
    proc.allow_errors = True

    proc.preprocess(nb, {"metadata": {"path": os.path.dirname(full_path)}})

    errors = []
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if output.output_type == "error":
                    errors.append(output)
                    # Print traceback so that it appears in the test log
                    print("".join(output["traceback"]), file=sys.stderr)
    return nb, errors


@pytest.fixture(params=NOTEBOOKS)
def notebook_name(request: Any) -> str:
    return request.param  # type: ignore


def test_notebook(notebook_name: str) -> None:
    _, errors = run_notebook(f"{NOTEBOOKS_DIR}/{notebook_name}")
    assert len(errors) == 0


if __name__ == "__main__":
    pytest.main(sys.argv)
