# PyCID: Causal Influence Diagrams library

This package implements causal influence diagrams and methods to analyze them, and is part of the
[Causal Incentives](https://causalincentives.com) project.

Building on [pgmpy](https://pgmpy.org/), pycid provides methods for
defining CIDs and MACIDs,
computing optimal policies and Nash equilibria,
studying the effects of interventions, and
checking graphical criteria for various types of incentives.

## Install
Create and activate
a [python virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) or
a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-envs).
Then install using:
```shell
python3 -m pip install pycid
```

PyCID requires python version 3.7 or greater.

## Basic usage

```python
# Import
import pycid

# Specify the nodes and edges of a simple CID
cid = pycid.CID([
    ('S', 'D'),   # add nodes S and D, and a link S -> D
    ('S', 'U'),   # add node U, and a link S -> U
    ('D', 'U'),   # add a link D -> U
    ],
    decisions=['D'],   # D is a decision node
    utilities=['U'])   # U is a utility node

# specify the causal relationships
cid.add_cpds(
    pycid.UniformRandomCPD('S', [-1, 1]),     # S is -1 or 1 with equal probability
    pycid.DecisionDomain('D', [-1, 1]),       # the permitted choices for D are -1 and 1
    pycid.FunctionCPD('U', lambda s, d: s*d)  # U is the product of S and D (arguments lowercase the variable names)
)

# Draw the result
cid.draw()
```

![image](./image.png "")

The [notebooks](./notebooks) provide many more examples, including
a [CID Basics Tutorial](./notebooks/CID_Basics_Tutorial.ipynb),
a [MACID Basics Tutorial](./notebooks/MACID_Basics_Tutorial.ipynb), and
a [CID Incentives Tutorial](./notebooks/CID_Incentives_Tutorial.ipynb).

To quickly get started using colab, make a copy of [this notebook](https://colab.research.google.com/drive/1QsITqn02c8zBfdOxaR0gVFcLQA_2UROu#scrollTo=GfupUXwM77Ti).

## Code overview

The code is structured into the following folders:
* [pycid/core](./pycid/core) contains methods and classes for specifying CID and MACID models,
  for finding and characterising types of paths in these models' graphs, and for
  computing optimal policies and Nash equilibria.
* [pycid/analyze](./pycid/analyze) has methods for analyzing different types of effects and interventions
as well as incentives in single-decision CIDs and reasoning patterns in MACIDs.
* [pycid/random](./pycid/random) has methods for generating random CIDs.
* [pycid/examples](./pycid/examples) has a range of pre-specified CIDs and MACIDs.
* [notebooks](./notebooks) has iPython notebooks illustrating the use of key methods.
* [tests](./tests) has unit tests for all public methods.

## Contributing
The project is developed at <https://github.com/causalincentives/pycid>.

### Install
First create and activate
a [python virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) or
a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-envs).
```shell
git clone https://github.com/causalincentives/pycid  # download the code
cd pycid
python3 -m pip install --editable .[test]
python3 -m pytest   # check that everything works
```

### Making Commits
Fast checks are set up as git pre-commit hooks.
To enable them, run:
```shell
pip3 install pre-commit
pre-commit install
```
They will run on every commit or can be run manually with `pre-commit run`.

Before committing to the master branch, please ensure that:
* The script [tests/check-code.sh](tests/check-code.sh) completes without error (you can add it as a pre-commit hook)
* Any new requirements are added to `setup.cfg`.
* Your functions have docstrings and types, and a unit test verifying that they work
* For notebooks, you have done "restart kernel and run all cells" before saving and committing
* Any documentation (such as this file) is up-to-date
