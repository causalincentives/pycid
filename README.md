# PyCID: Causal Influence Diagrams library

This package implements causal influence diagrams and methods to analyze them, and is part of the
[Causal Incentives](https://causalincentives.com) project.

Building on [pgmpy](https://pgmpy.org/) and [NetworkX](https://networkx.org/), pycid provides methods for
defining CBNs, CIDs and MACIDs,
computing optimal policies in CIDs, pure and mixed Nash equilibria in multi-agent CIDs,
studying the effects of interventions, and
checking graphical criteria for various types of incentives.

## News

Version 0.7 *breaks backwards compatibility* by requiring CPD arguments to match the case of the parent nodes.
To update your code to the latest version, simply change the case of the arguments, as illustrated [here](https://github.com/causalincentives/pycid/commit/e50ee06b7eafac63fe7c9471764c9c5774fc743b).
Alternatively, stick to version 0.2.8.

## Install
Create and activate
a [python virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) or
a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-envs).
Then install using:
```shell
python3 -m pip install pycid
```

PyCID requires python version 3.8 or greater.

## Basic usage

```python
# Import
import pycid

# Specify the nodes and edges of a simple CID
cid = pycid.CID([
    ('S', 'D'),  # add nodes S and D, and a link S -> D
    ('S', 'U'),  # add node U, and a link S -> U
    ('D', 'U'),  # add a link D -> U
],
    decisions=['D'],  # D is a decision node
    utilities=['U'])  # U is a utility node

# specify the causal relationships with CPDs using keyword arguments
cid.add_cpds(S = pycid.discrete_uniform([-1, 1]), # S is -1 or 1 with equal probability
             D=[-1, 1], # the permitted action choices for D are -1 and 1
             U=lambda S, D: S * D) # U is the product of S and D (argument names match parent names)

# Draw the result
cid.draw()
```

![image](./image.png "")

The [notebooks](./notebooks) provide many more examples, including:
* [CBN Tutorial](https://colab.research.google.com/github/causalincentives/pycid/blob/master/notebooks/CBN_Tutorial.ipynb) shows how to specify the structure and (causal) relationships between nodes, and ask simple queries.
* [CID tutorial](https://colab.research.google.com/github/causalincentives/pycid/blob/master/notebooks/CID_Basics_Tutorial.ipynb) adds special decision and utility nodes for one agent, and how to compute optimal policies.
* [MACID tutorial](https://colab.research.google.com/github/causalincentives/pycid/blob/master/notebooks/MACID_Basics_Tutorial.ipynb) covers methods for handling multiple agents, including finding subgames and Nash equilibria.
* [Incentive Analysis tutorial](https://colab.research.google.com/github/causalincentives/pycid/blob/master/notebooks/CID_Incentives_Tutorial.ipynb) illustrates various methods for analyzing the incentives of agents.

The above notebooks links all open in Colab, and can be run
directly in the browser with no further setup or installation required.

## Code overview

The code is structured into the following folders:
* [pycid/core](./pycid/core) contains methods and classes for specifying CBN, CID and MACID models,
  for finding and characterising types of paths in these models' graphs, and for
  computing optimal policies and Nash equilibria.
* [pycid/analyze](./pycid/analyze) has methods for analyzing different types of effects and interventions
as well as incentives in single-decision CIDs and reasoning patterns in MACIDs.
* [pycid/random](./pycid/random) has methods for generating random CIDs.
* [pycid/examples](./pycid/examples) has a range of pre-specified CBNs, CIDs and MACIDs.
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

## Citing
Please use the following BibTeX entry for citing `PyCID` in your research:

```
@InProceedings{ james_fox-proc-scipy-2021,
  author    = { {J}ames {F}ox and {T}om {E}veritt and {R}yan {C}arey and {E}ric {L}anglois and {A}lessandro {A}bate and {M}ichael {W}ooldridge },
  title     = { {P}y{C}{I}{D}: {A} {P}ython {L}ibrary for {C}ausal {I}nfluence {D}iagrams },
  booktitle = { {P}roceedings of the 20th {P}ython in {S}cience {C}onference },
  pages     = { 43 - 51 },
  year      = { 2021 },
  editor    = { {M}eghann {A}garwal and {C}hris {C}alloway and {D}illon {N}iederhut and {D}avid {S}hupe },
  doi       = {10.25080/majora-1b6fd038-008}
}
```
