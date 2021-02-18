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
cid = CID([('S', 'D'), ('S', 'U'), ('D', 'U')],
          decision_nodes=['D'],
          utility_nodes=['U'])

# specify the causal relationships
cpd_s = UniformRandomCPD('S', [-1, 1])
cpd_d = DecisionDomain('D', [-1, 1])
cpd_u = FunctionCPD('U', lambda s, d: s*d, evidence=['S', 'D'])
cid.add_cpds(cpd_d, cpd_s, cpd_u)

# Draw the result
cid.draw()
```

![image](./image.png "")

The [notebooks](./notebooks) provide many more examples.

## Code overview

The code is structured into 5 folders:
* [core](./core) contains methods and classes for specifying CID and MACID models, 
  for finding and characterising types of paths in these models' graphs, and for 
  computing optimal policies and Nash equilibria.
* [examples](./examples) has a range of pre-specified CIDs and MACIDs, 
  as well as methods for generating random ones.
* [analyze](./analyze) has methods for analyzing different types of effects and interventions
as well as incentives in single-decision CIDs and reasoning patterns in MACIDs.
* [notebooks](./notebooks) has iPython notebooks illustrating the use of key methods:
  * [CID_Basics_Tutorial](./notebooks/CID_Basics_Tutorial) introduces how to instantiate, plot,
  and perform basic methods on CIDs.
  * [MACID_Basics_Tutorial](./notebooks/CID_Basics_Tutorial) introduces how to instantiate, plot,
  and perform basic methods on CIDs.
  * [CID_Incentives_Tutorial](./notebooks/CID_Incentives_Tutorial) shows how to find which nodes
  in a single decision CID amits incentives: value of control (direct and indirect), value of information, response incentive, and instrumental control incentives.
  * [Reasoning_Patterns_Tutorial](./notebooks/CID_Incentives_Tutorial) shows how to find which decison nodes
  reasoning patterns in a MACID.
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
* All tests pass: run `python3 -m unittest` from the root directory
* Your code does not generate unnecessary `pylint` warnings 
  (some are okay, if fixing them would be hard)
* Any added requirements are added to Requirements.txt
* Your functions have docstrings and types, and a unit test verifying that they work
* You have chosen "restart kernel and run all cells" before saving and committing a notebook
* Any documentation (such as this file) is up-to-date