from pgmpy.factors.base import BaseFactor
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import (
    TabularCPD,
    JointProbabilityDistribution,
    DiscreteFactor,
)
from pgmpy.factors.continuous import ContinuousFactor
#TODO: add typing


class NullCPD(BaseFactor):
    def __init__(self, variable, variable_card):
        self.variable = variable
        self.variable_card = variable_card #is this correct?
        self.cardinality = [variable_card]
        self.variables = [self.variable]

    def scope(self):
        return [self.variable]

    def to_factor(self):
        return self


class CID(BayesianModel):
    def __init__(self, ebunch=None):
        super(CID, self).__init__(ebunch=ebunch)

    def check_sufficient_recall(self):
        pass

    def solve(self, decision_name):
        #returns an optimal cpd for a given decision
        pass

    def expected_utility(self, cpd_dict):
        #input a cpd for each decision
        pass

    def add_cpds(self, *cpds):
        for cpd in cpds:
            if not isinstance(cpd, (TabularCPD, ContinuousFactor, NullCPD)):
                raise ValueError("Only TabularCPD, ContinuousFactor, or NullCPD can be added.")

            if set(cpd.scope()) - set(cpd.scope()).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning(
                        "Replacing existing CPD for {var}".format(var=cpd.variable)
                    )
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def check_model(self, allow_null=True):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if cpd is None:
                raise ValueError("No CPD associated with {}".format(node))
            elif isinstance(cpd, (TabularCPD, ContinuousFactor)):
                evidence = cpd.get_evidence()
                parents = self.get_parents(node)
                if set(evidence if evidence else []) != set(parents if parents else []):
                    raise ValueError(
                        "CPD associated with {node} doesn't have "
                        "proper parents associated with it.".format(node=node)
                    )
                if not cpd.is_valid_cpd():
                    raise ValueError(
                        "Sum or integral of conditional probabilites for node {node}"
                        " is not equal to 1.".format(node=node)
                    )
            elif isinstance(cpd, (NullCPD)):
                if not allow_null:
                    raise ValueError(
                        "CPD associated with {node} is nullcpd".format(node=node)
                    )
        return True

    #def _get_local_policies(self, decision_name):
    #    #returns a list of all possible CPDs
    #    pass

    def _get_sp_policy(self, decision_name):
        policy = {}
        for context in contexts:
            self._get_best_act(self, decision_name, context)
            policy[context] = act
        return TabularCPD(policy)

    def _get_best_act(self, decision_name, context):
        utilities = []
        for act in acts:
            pdf = BeliefPropagation(self).query(act)
            utilities.append(pdf.expectation())
        return acts[np.argmax(utilities)]



def get_minimal_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([('A', 'B')])
    cpd = TabularCPD('B',2,[[1., 0.], [0., 1.]], evidence=['A'], evidence_card = [2])
    #import ipdb; ipdb.set_trace()
    cid.add_cpds(NullCPD('A', 2), cpd)
    return cid
    
