import os
import sys
import unittest
from typing import Any, List, Tuple

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


class TestNotebooks(unittest.TestCase):
    def test_fairness_notebook(self) -> None:
        _, errors = run_notebook("notebooks/fairness.ipynb")
        self.assertEqual(len(errors), 0)

    def test_cid_basics_tutorial_notebook(self) -> None:
        _, errors = run_notebook("notebooks/CID_Basics_Tutorial.ipynb")
        self.assertEqual(len(errors), 0)

    def test_cid_incentives_tutorial_notebook(self) -> None:
        _, errors = run_notebook("notebooks/CID_Incentives_Tutorial.ipynb")
        self.assertEqual(len(errors), 0)

    def test_generate_cid_notebook(self) -> None:
        _, errors = run_notebook("notebooks/generate_cid.ipynb")
        self.assertEqual(len(errors), 0)

    def test_macid_basics_tutorial_notebook(self) -> None:
        _, errors = run_notebook("notebooks/MACID_Basics_Tutorial.ipynb")
        self.assertEqual(len(errors), 0)

    def test_reasoning_patterns_tutorial_notebook(self) -> None:
        _, errors = run_notebook("notebooks/Reasoning_Patterns_Tutorial.ipynb")
        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    pytest.main(sys.argv)
