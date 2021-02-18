# Causal Influence Diagram Representation

This package implements causal influence diagrams and methods to analyze them, and is part of the
[Causal Incentives](https://causalincentives.com) project.

Building on [pgmpy](https://pgmpy.org/), pycid provides methods for 
defining CIDs and MACIDs, 
computing optimal policies and Nash equilibria,
studying the effects of interventions, and
checking graphical criteria for various types of incentives. 

## Basic usage

```python
# Import
from core.cid import CID
from core.cpd import UniformRandomCPD, DecisionDomain, FunctionCPD

# Specify the nodes and edges of a simple CID
cid = CID([('S', 'D'),   # add nodes S and D, and a link S -> D
           ('S', 'U'),   # add node U, and a link S -> U
           ('D', 'U')],  # add a link D -> U
          decision_nodes=['D'],  # D is a decision node
          utility_nodes=['U'])   # U is a utility node

# specify the causal relationships
cid.add_cpds(
    UniformRandomCPD('S', [-1, 1]),  # S is -1 or 1 with equal probability
    DecisionDomain('D', [-1, 1]),  # the permitted choices for D are -1 and 1
    FunctionCPD('U', lambda s, d: s*d, evidence=['S', 'D']) # U is the product of S and D
)

# Draw the result
cid.draw()
```

![image](./image.png "")

The [notebooks](./notebooks) provide many more examples, including
a [CID Basics Tutorial](./notebooks/CID_Basics_Tutorial.ipynb), 
a [MACID Basics Tutorial](./notebooks/MACID_Basics_Tutorial.ipynb), and
a [CID Incentives Tutorial](./notebooks/CID_Incentives_Tutorial.ipynb).

## Code overview

The code is structured into 5 folders:
* [core](./core) contains methods and classes for specifying CID and MACID models, 
  for finding and characterising types of paths in these models' graphs, and for 
  computing optimal policies and Nash equilibria.
* [examples](./examples) has a range of pre-specified CIDs and MACIDs, 
  as well as methods for generating random ones.
* [analyze](./analyze) has methods for analyzing different types of effects and interventions
as well as incentives in single-decision CIDs and reasoning patterns in MACIDs.
* [notebooks](./notebooks) has iPython notebooks illustrating the use of key methods.
* [test](./test) has unit tests for all public methods.

## Installation and setup

Given that you have Python 3.7 or later, git, and jupyter, 
you can download and setup pycid via:

```shell
git clone https://github.com/causalincentives/pycid  # download the code
cd pycid
pip3 install -r Requirements.txt  # install required python packages
python3 -m unittest   # check that everything works
```

## Contributing

Before committing to the master branch, please ensure that:
* The script [test/check-code.sh](test/check-code.sh) completes without error (you can add it as a pre-commit hook)
* Any new requirements are added to Requirements.txt
* Your functions have docstrings and types, and a unit test verifying that they work
* For notebooks, you have done "restart kernel and run all cells" before saving and committing 
* Any documentation (such as this file) is up-to-date
