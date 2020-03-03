# Causal Influence Diagram Representation

This project implements causal influence diagrams, 
as studied in a number of recent papers and blog posts.

* [The Incentives that Shape Behavior](https://arxiv.org/abs/2001.07118)
* [Reward Tampering Problems and Solutions](https://medium.com/@deepmindsafetyresearch/designing-agent-incentives-to-avoid-reward-tampering-4380c1bb6cd)
* [Understanding Agent Incentives using Causal Influence Diagrams](https://medium.com/@deepmindsafetyresearch/understanding-agent-incentives-with-causal-influence-diagrams-7262c2512486)
* [Modeling AGI Safety Frameworks with Causal Influence Diagrams](https://arxiv.org/abs/1906.08663)
* TODO: Add some old ones

The key class is CID.py, which extends [BayesianModel](http://pgmpy.org/models.html) from the well-established Python graphical models library [pgmpy](http://pgmpy.org).

Just like BayesianModel, it can be flexibly initialized in a number of ways. 
It requires two additional parameters that specify the decision nodes and the utility nodes of the model.

Building on pgmpy, it provides methods for computing
* The expected utility (of a decision) (in some context)
* The set of optimal decisions in a given context
* A check for [sufficient recall](http://people.csail.mit.edu/milch/papers/geb-maid.pdf) / [solubility](https://arxiv.org/abs/1301.3881)
* Optimal policies for all decision nodes if sufficient recall is satisfied.

A number of example models are provided in the folder models.

### Setup

Install the required packages `pip3 install -r requirements.txt` and run `python3 tests/cid_test.py` to check that everything works.